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
    bond_type: str = Field(description="Performance Bond / Payment Bond / Letter of Credit / Cash / Certificate of Deposit")
    forms_provided: str = Field(description="Yes/No - whether exhibits/appendices included")
    adjustable_amount: str = Field(description="Yes/No - whether subject to mid-term increases")
    cancellation_terms: str = Field(description="Verbatim cancellation language")
    cancellation_notice_period: str = Field(description="e.g. '30 Days'")
    effective_duration: str = Field(description="Term or coverage period")
    forfeiture_language: str = Field(description="Verbatim forfeiture clause or 'Not Present'")
    quick_payment_terms: str = Field(description="Payment required within 30 days or less - Yes/No and details")
    efficiency_guarantees: str = Field(description="Performance, quality, or output accountability - details or 'Not Present'")
    enforcement_mechanisms: str = Field(description="Claims process, notices, time limits, triggering events")

class StatuteRef(BaseModel):
    citation: str
    name: str
    verbatim_text: Optional[str] = Field(description="Excerpt from official source")
    plain_summary: Optional[str] = Field(description="Simple explanation")
    source_link: Optional[str] = Field(description="URL to .gov or legal source")
    found: bool = True

class UnderwritingOpinion(BaseModel):
    risk_state: str = Field(description="State where bond is issued")
    obligee: str = Field(description="Name of obligee")
    bond_description: str = Field(description="Purpose/type of bond")
    sfaa_descriptor: Optional[str] = Field(default="N/A", description="SFAA No. or Descriptor Code if available")
    penalty: str = Field(description="Penal sum / bond amount")
    statutory_references: List[str] = Field(description="List of statutes/rules cited")
    # Legal Opinion Structured Components
    who_must_post: str = Field(description="Who is required to post the bond")
    statutory_definitions: str = Field(description="Relevant definitions from statutes")
    exemptions: str = Field(description="Who is exempt from bond requirement")
    bond_guarantees: str = Field(description="What the bond guarantees/covers")
    deadlines_cycles: str = Field(description="Licensing/bonding deadlines and expiration cycles")
    claims_history: str = Field(description="Past claims experience or loss ratio if known")
    alternative_instruments: str = Field(description="Whether bond can be replaced by other financial instruments")
    # Cancellation
    cancellation_verbatim: str = Field(description="Exact cancellation clause language")
    cancellation_summary: str = Field(description="Plain language summary of notice period")
    # Recommendation
    recommendation: str = Field(description="Approve / Decline / Refer")

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
    2. Key Risk Clauses - extract ALL of the following (use "Not Present" if clause doesn't exist):
       - security_required: Yes/No and type details
       - bond_type: Performance Bond / Payment Bond / Letter of Credit / Cash / Certificate of Deposit
       - forms_provided: Yes/No - are exhibits/appendices included?
       - adjustable_amount: Yes/No - can amount increase mid-term?
       - cancellation_terms: The EXACT verbatim cancellation language
       - cancellation_notice_period: e.g. "30 Days"
       - effective_duration: Term or coverage period
       - forfeiture_language: Exact forfeiture clause or "Not Present"
       - quick_payment_terms: Payment within 30 days or less? Details or "Not Present"
       - efficiency_guarantees: Performance/quality/output accountability or "Not Present"
       - enforcement_mechanisms: Claims process, notice requirements, time limits, triggering events
    3. A list of ALL legal statutes, codes, or regulations explicitly cited (e.g., "California Civil Code 1234", "Public Contract Code").
    
    Return JSON matching the structure:
    {
        "parties": { "principal_name": "...", "obligee_name": "...", "surety_name": "...", "penal_sum": "..." },
        "risks": { 
            "security_required": "...", 
            "bond_type": "...", 
            "forms_provided": "...",
            "adjustable_amount": "...",
            "cancellation_terms": "...", 
            "cancellation_notice_period": "...", 
            "effective_duration": "...",
            "forfeiture_language": "...", 
            "quick_payment_terms": "...",
            "efficiency_guarantees": "...",
            "enforcement_mechanisms": "..."
        },
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

    # Search for each statute individually to ensure grounding
    researched = []
    
    for citation in citations:
        print(f"Searching for: {citation}")
        
        search_prompt = f"""
        Find information about this legal statute: "{citation}"
        
        Use Google Search to find the official statute text.
        
        Return ONLY information you can verify from search results. If you cannot find it, set "found": false.
        
        Return JSON:
        {{
            "citation": "{citation}",
            "name": "Official name of the statute/act",
            "verbatim_text": "2-3 sentence excerpt from the official source you found",
            "plain_summary": "1 sentence plain summary",
            "source_link": "The actual URL from search results",
            "found": true or false
        }}
        """
        
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=search_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            
            # Check if response has grounding metadata
            print(f"Response for {citation}:")
            print(f"  Text: {response.text[:200]}...")
            
            # Check if search was actually used
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    print(f"  ✓ Has grounding metadata")
                else:
                    print(f"  ⚠ NO grounding metadata - response may be hallucinated")
            
            data = json.loads(response.text)
            
            # Validate: if found=true but no source_link, mark as suspicious
            if data.get('found') and not data.get('source_link'):
                print(f"  ⚠ WARNING: Marked as found but no source link provided")
                data['found'] = False
                data['verbatim_text'] = "Statute identified but could not retrieve official text"
                data['plain_summary'] = "Unable to verify statute details"
            
            researched.append(StatuteRef(**data))
            
        except Exception as e:
            print(f"Error researching {citation}: {e}")
            researched.append(StatuteRef(
                citation=citation,
                name="Research Error",
                verbatim_text="Unable to retrieve statute information",
                plain_summary="Statute lookup failed",
                source_link=None,
                found=False
            ))
    
    return {"researched_statutes": researched}

def opinion_node(state: BondState):
    """Step 3: Synthesize the Underwriting Opinion."""
    print(f"--- BOND AGENT: SYNTHESIZING OPINION ---")
    
    # Prepare context
    parties_txt = state.parties.model_dump_json() if state.parties else ""
    risks_txt = state.risks.model_dump_json() if state.risks else ""
    statutes_txt = json.dumps([s.model_dump() for s in state.researched_statutes])

    prompt = f"""
    Generate a Commercial Underwriting Opinion for this bond in the exact structured format required.
    
    Data Available:
    Parties: {parties_txt}
    Risks: {risks_txt}
    Statutory Research: {statutes_txt}
    
    Generate JSON with ALL fields:
    {{
        "risk_state": "State inferred from obligee/statutes",
        "obligee": "Name of obligee from parties data",
        "bond_description": "Brief purpose/type of bond",
        "sfaa_descriptor": "SFAA No. or Descriptor Code if found, otherwise 'N/A'",
        "penalty": "The penal sum amount",
        "statutory_references": ["List", "of", "statute", "citations"],
        
        "who_must_post": "Who is required to post the bond - be specific",
        "statutory_definitions": "Relevant definitions from the statutes cited",
        "exemptions": "Who is exempt from the bond requirement - or 'None identified'",
        "bond_guarantees": "What the bond specifically guarantees or covers",
        "deadlines_cycles": "Licensing/bonding deadlines and expiration cycles",
        "claims_history": "Past claims experience or loss ratio if known from research, otherwise 'No data available'",
        "alternative_instruments": "Whether bond can be replaced by cash, LOC, CD, etc.",
        
        "cancellation_verbatim": "Exact cancellation clause from the bond",
        "cancellation_summary": "Plain language: notice period and rights upon cancellation",
        
        "recommendation": "Approve / Decline / Refer - with brief reasoning"
    }}
    
    Be thorough and specific. Use the researched statutes to inform the legal opinion components.
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
