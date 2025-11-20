import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types

# ==========================================
# 1. CONFIGURATION
# ==========================================

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "" 

# Primary client for extraction & opinion
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-3-pro-preview"

# ==========================================
# 2. DATA MODELS
# ==========================================

class BondParties(BaseModel):
    principal_name: str = Field(description="Name of the Principal")
    obligee_name: str = Field(description="Name of the Obligee")
    surety_name: Optional[str] = Field(default=None)
    penal_sum: str = Field(description="The formatted penal sum amount (e.g. $1,000,000)")

class RiskClauses(BaseModel):
    security_required: str = Field(description="Yes/No and details")
    bond_type: str = Field(description="Performance, Payment, etc.")
    cancellation_terms: str = Field(description="Verbatim cancellation language")
    cancellation_notice_period: str = Field(description="e.g. '30 Days'")
    forfeiture_language: str = Field(description="Yes/No and details")
    duration: str = Field(description="Term or coverage period")
    risk_level: str = Field(description="Low, Medium, High based on terms")
    payment_terms: str = Field(description="Quick payment terms if any")
    enforcement: str = Field(description="Claims process or triggering events")

class StatuteRef(BaseModel):
    citation: str
    name: str
    verbatim_text: Optional[str] = Field(description="Excerpt from official source")
    plain_summary: Optional[str] = Field(description="Simple explanation")
    source_link: Optional[str] = Field(description="URL to .gov or legal source")
    found: bool = True

class UnderwritingOpinion(BaseModel):
    risk_state: str
    bond_description: str
    legal_opinion_text: str = Field(description="The synthesized legal opinion paragraphs")
    recommendation: str = Field(description="Approve, Decline, or Refer")
    cancellation_summary: str

class BondState(BaseModel):
    file_path: str
    # Step 1 Data
    parties: Optional[BondParties] = None
    risks: Optional[RiskClauses] = None
    identified_citations: List[str] = []
    # Step 2 Data
    researched_statutes: List[StatuteRef] = []
    # Step 3 Data
    opinion: Optional[UnderwritingOpinion] = None
    # Final Output
    final_json: Dict[str, Any] = {}

# ==========================================
# 3. NODES
# ==========================================

def extraction_node(state: BondState):
    """Step 1: Extract hard data and identify citations."""
    print(f"--- BOND AGENT: EXTRACTING DATA ---")
    
    try:
        with open(state.file_path, "rb") as f:
            file_bytes = f.read()
    except:
        return {}

    prompt = """
    Analyze this Bond Form/Contract. Extract the following:
    1. The Parties (Principal, Obligee, Surety) and Penal Sum.
    2. Key Risk Clauses (Cancellation terms, notice periods, forfeiture, duration, enforcement).
    3. A list of ALL legal statutes, codes, or regulations explicitly cited (e.g., "California Civil Code 1234", "Public Contract Code").
    
    Return JSON matching the structure:
    {
        "parties": { "principal_name": "...", "obligee_name": "...", "surety_name": "...", "penal_sum": "..." },
        "risks": { "security_required": "...", "bond_type": "...", "cancellation_terms": "...", "cancellation_notice_period": "...", "forfeiture_language": "...", "duration": "...", "risk_level": "...", "payment_terms": "...", "enforcement": "..." },
        "identified_citations": ["Code A", "Regulation B"]
    }
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        
        data = json.loads(response.text)
        
        return {
            "parties": BondParties(**data.get("parties", {})),
            "risks": RiskClauses(**data.get("risks", {})),
            "identified_citations": data.get("identified_citations", [])
        }
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {}

def research_node(state: BondState):
    """Step 2: Use Google Search to look up the identified statutes."""
    print(f"--- BOND AGENT: RESEARCHING STATUTES ---")
    
    citations = state.identified_citations
    if not citations:
        return {"researched_statutes": []}

    # Construct a search prompt for the tool
    search_prompt = f"""
    I have a list of legal citations found in a bond form: {json.dumps(citations)}.
    
    For EACH citation, I need you to find the official legal text using Google Search.
    
    Return a JSON list:
    [
        {{
            "citation": "The citation code",
            "name": "The name of the act/statute",
            "verbatim_text": "A short 2-3 sentence excerpt from the official source",
            "plain_summary": "A 1 sentence plain english summary",
            "source_link": "The URL to the .gov or legal database found",
            "found": true
        }}
    ]
    
    If you cannot find a specific statute, set "found": false.
    """

    try:
        # Enable Google Search Tool
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=search_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        
        raw_data = response.text
        # Sometimes the search tool might return text mixed with grounding; ensure we parse JSON
        try:
            data = json.loads(raw_data)
        except:
            # Fallback if it returns a list directly
            if isinstance(raw_data, list):
                data = raw_data
            else:
                data = []

        statutes = [StatuteRef(**s) for s in data]
        return {"researched_statutes": statutes}

    except Exception as e:
        print(f"Research Error: {e}")
        return {"researched_statutes": []}

def opinion_node(state: BondState):
    """Step 3: Synthesize the Underwriting Opinion."""
    print(f"--- BOND AGENT: SYNTHESIZING OPINION ---")
    
    # Prepare context
    parties_txt = state.parties.model_dump_json() if state.parties else ""
    risks_txt = state.risks.model_dump_json() if state.risks else ""
    statutes_txt = json.dumps([s.model_dump() for s in state.researched_statutes])

    prompt = f"""
    Generate a Commercial Underwriting Opinion for this bond.
    
    Data Available:
    Parties: {parties_txt}
    Risks: {risks_txt}
    Statutory Research: {statutes_txt}
    
    Generate JSON:
    {{
        "risk_state": "State inferred from obligee/statutes",
        "bond_description": "Brief purpose",
        "legal_opinion_text": "Detailed legal opinion covering: Who posts, definitions, exemptions, guarantees, deadlines, loss ratio history (if known/searchable), and replacement options.",
        "recommendation": "Approve / Decline / Refer",
        "cancellation_summary": "Plain language summary of cancellation notice period and rights."
    }}
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
        )
        
        data = json.loads(response.text)
        opinion = UnderwritingOpinion(**data)
        
        # Construct Final Payload for Frontend
        final_payload = {
            "type": "BOND",
            "step_1": {
                "parties": state.parties.model_dump() if state.parties else {},
                "risks": state.risks.model_dump() if state.risks else {}
            },
            "step_2": {
                "statutes": [s.model_dump() for s in state.researched_statutes]
            },
            "step_3": {
                "opinion": opinion.model_dump()
            }
        }
        
        return {"opinion": opinion, "final_json": final_payload}

    except Exception as e:
        print(f"Opinion Error: {e}")
        return {"final_json": {"error": str(e)}}

# ==========================================
# 4. GRAPH SETUP
# ==========================================

workflow = StateGraph(BondState)
workflow.add_node("extract", extraction_node)
workflow.add_node("research", research_node)
workflow.add_node("opinion", opinion_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "research")
workflow.add_edge("research", "opinion")
workflow.add_edge("opinion", END)

app = workflow.compile()
