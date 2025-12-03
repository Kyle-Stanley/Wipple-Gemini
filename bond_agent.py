import os
import json
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# Model client abstraction
from model_client import get_client, MetricsTracker, DEFAULT_MODEL

# ==========================================
# 1. CONFIGURATION
# ==========================================

# No longer need direct client initialization - handled by model_client

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
    bond_guarantees: str = Field(description="What the bond specifically guarantees or covers")
    deadlines_cycles: str = Field(description="Licensing/bonding deadlines and expiration cycles")
    claims_history: str = Field(description="Past claims experience or loss ratio if known from research, otherwise 'No data available'")
    alternative_instruments: str = Field(description="Whether bond can be replaced by other financial instruments")
    # Cancellation
    cancellation_verbatim: str = Field(description="Exact cancellation clause language")
    cancellation_summary: str = Field(description="Plain language summary of notice period")
    # Recommendation
    recommendation: str = Field(description="Approve / Decline / Refer")

class BondState(BaseModel):
    file_path: str
    model_name: str = DEFAULT_MODEL
    # Metrics tracker stored as dict for Pydantic serialization
    metrics_data: Dict[str, Any] = {}
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
    print(f"--- USING MODEL: {state.model_name} ---")
    
    # Initialize metrics tracker
    tracker = MetricsTracker(model_name=state.model_name)
    client = get_client()
    
    try:
        with open(state.file_path, "rb") as f:
            file_bytes = f.read()
    except:
        return {"metrics_data": tracker.get_metrics()}

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
        response = client.generate_content(
            prompt=prompt,
            model_name=state.model_name,
            pdf_bytes=file_bytes,
            response_mime_type="application/json",
            tracker=tracker,
        )
        
        data = json.loads(response.text)
        
        return {
            "parties": BondParties(**data.get("parties", {})),
            "risks": RiskClauses(**data.get("risks", {})),
            "identified_citations": data.get("identified_citations", []),
            "metrics_data": tracker.get_metrics(),
        }
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {"metrics_data": tracker.get_metrics()}

def research_node(state: BondState):
    """Step 2: Use Google Search to look up the identified statutes."""
    print(f"--- BOND AGENT: RESEARCHING STATUTES ---")
    print(f"--- USING MODEL: {state.model_name} ---")
    
    # Rebuild tracker from stored metrics
    prev_metrics = state.metrics_data or {}
    tracker = MetricsTracker(model_name=state.model_name)
    tracker.total_input_tokens = prev_metrics.get("tokens", {}).get("input", 0)
    tracker.total_output_tokens = prev_metrics.get("tokens", {}).get("output", 0)
    tracker.call_count = prev_metrics.get("api_calls", 0)
    
    client = get_client()
    
    citations = state.identified_citations
    if not citations:
        return {"researched_statutes": [], "metrics_data": tracker.get_metrics()}

    # Search for each statute individually to ensure grounding
    researched = []
    
    # Note: Google Search grounding only works with Gemini models
    # For Anthropic models, we'll skip the search and mark as not found
    is_google_model = state.model_name.startswith("gemini")
    
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
            if is_google_model:
                # Use Google Search grounding
                response = client.generate_content(
                    prompt=search_prompt,
                    model_name=state.model_name,
                    response_mime_type="application/json",
                    use_google_search=True,
                    tracker=tracker,
                )
            else:
                # For non-Google models, provide context that search isn't available
                no_search_prompt = f"""
                I need information about this legal statute: "{citation}"
                
                Based on your knowledge, provide what you know about this statute.
                If you're not certain, set "found": false.
                
                Return JSON:
                {{
                    "citation": "{citation}",
                    "name": "Official name of the statute/act if known",
                    "verbatim_text": "General description of what this statute covers",
                    "plain_summary": "1 sentence plain summary",
                    "source_link": null,
                    "found": false
                }}
                
                Note: Set found to false since we cannot verify with live search.
                """
                response = client.generate_content(
                    prompt=no_search_prompt,
                    model_name=state.model_name,
                    response_mime_type="application/json",
                    tracker=tracker,
                )
            
            # Robust JSON parsing
            try:
                data = json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback: Try to find JSON within the response text if it's mixed with markdown
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    raise Exception("No valid JSON found in response")

            # Default values for missing fields
            citation_data = {
                "citation": citation,
                "name": data.get("name", "Unknown"),
                "verbatim_text": data.get("verbatim_text"),
                "plain_summary": data.get("plain_summary"),
                "source_link": data.get("source_link"),
                "found": data.get("found", False)
            }

            # Validate: if found=true but no source_link, mark as suspicious
            if citation_data['found'] and not citation_data['source_link']:
                print(f"  WARNING: Marked as found but no source link provided")
                citation_data['found'] = False
                citation_data['verbatim_text'] = "Statute identified but could not retrieve official text"
                citation_data['plain_summary'] = "Unable to verify statute details"
            
            researched.append(StatuteRef(**citation_data))
            
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
    
    return {"researched_statutes": researched, "metrics_data": tracker.get_metrics()}

def opinion_node(state: BondState):
    """Step 3: Synthesize the Underwriting Opinion."""
    print(f"--- BOND AGENT: SYNTHESIZING OPINION ---")
    print(f"--- USING MODEL: {state.model_name} ---")
    
    # Rebuild tracker from stored metrics
    prev_metrics = state.metrics_data or {}
    tracker = MetricsTracker(model_name=state.model_name)
    tracker.total_input_tokens = prev_metrics.get("tokens", {}).get("input", 0)
    tracker.total_output_tokens = prev_metrics.get("tokens", {}).get("output", 0)
    tracker.call_count = prev_metrics.get("api_calls", 0)
    
    client = get_client()
    
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
        "exemptions": "Who is exempt from bond requirement - or 'None identified'",
        "bond_guarantees": "What the bond specifically guarantees or covers",
        "deadlines_cycles": "Licensing/bonding deadlines and expiration cycles",
        "claims_history": "Past claims experience or loss ratio if known from research, otherwise 'No data available'",
        "alternative_instruments": "Whether bond can be replaced by other financial instruments",
        
        "cancellation_verbatim": "Exact cancellation clause from the bond",
        "cancellation_summary": "Plain language summary of notice period",
        
        "recommendation": "Approve / Decline / Refer - with brief reasoning"
    }}
    
    Be thorough and specific. Use the researched statutes to inform the legal opinion components.
    """

    try:
        response = client.generate_content(
            prompt=prompt,
            model_name=state.model_name,
            response_mime_type="application/json",
            temperature=0.2,
            tracker=tracker,
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
            },
            "metrics": tracker.get_metrics(),
        }
        
        return {"opinion": opinion, "final_json": final_payload, "metrics_data": tracker.get_metrics()}

    except Exception as e:
        print(f"Opinion Error: {e}")
        return {"final_json": {"error": str(e), "metrics": tracker.get_metrics()}, "metrics_data": tracker.get_metrics()}

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
