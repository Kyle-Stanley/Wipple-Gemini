# ======================
# bond_agent.py (UPDATED)
# ======================
from __future__ import annotations

import json
import os
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_client import (
    get_client,
    MetricsTracker,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    ModelProvider,
    parse_json_safely,
)

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

    who_must_post: str = Field(description="Who is required to post the bond")
    statutory_definitions: str = Field(description="Relevant definitions from statutes")
    exemptions: str = Field(description="Who is exempt from bond requirement")
    bond_guarantees: str = Field(description="What the bond specifically guarantees or covers")
    deadlines_cycles: str = Field(description="Licensing/bonding deadlines and expiration cycles")
    claims_history: str = Field(description="Past claims experience or loss ratio if known from research, otherwise 'No data available'")
    alternative_instruments: str = Field(description="Whether bond can be replaced by other financial instruments")

    cancellation_verbatim: str = Field(description="Exact cancellation clause language")
    cancellation_summary: str = Field(description="Plain language summary of notice period")

    recommendation: str = Field(description="Approve / Decline / Refer")

class BondState(BaseModel):
    file_path: str
    model_name: str = DEFAULT_MODEL

    metrics_data: Dict[str, Any] = Field(default_factory=dict)

    parties: Optional[BondParties] = None
    risks: Optional[RiskClauses] = None
    identified_citations: List[str] = Field(default_factory=list)

    researched_statutes: List[StatuteRef] = Field(default_factory=list)

    opinion: Optional[UnderwritingOpinion] = None
    final_json: Dict[str, Any] = Field(default_factory=dict)

# ==========================================
# 3. NODES
# ==========================================

def extraction_node(state: BondState):
    print(f"--- BOND AGENT: EXTRACTING DATA ---")
    print(f"--- USING MODEL: {state.model_name} ---")

    tracker = MetricsTracker(model_name=state.model_name)
    client = get_client()

    try:
        with open(state.file_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        print(f"File Read Error: {e}")
        return {"metrics_data": tracker.get_metrics()}

    prompt = """
    Analyze this Bond Form/Contract. Extract the following:
    1. The Parties (Principal, Obligee, Surety) and Penal Sum.
    2. Key Risk Clauses - extract ALL of the following (use "Not Present" if clause doesn't exist):
       - security_required
       - bond_type
       - forms_provided
       - adjustable_amount
       - cancellation_terms (EXACT verbatim)
       - cancellation_notice_period
       - effective_duration
       - forfeiture_language
       - quick_payment_terms
       - efficiency_guarantees
       - enforcement_mechanisms
    3. A list of ALL legal statutes, codes, or regulations explicitly cited.

    Return JSON matching the structure:
    {
        "parties": { "principal_name": "...", "obligee_name": "...", "surety_name": "...", "penal_sum": "..." },
        "risks": { ... },
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
            system_prompt="You are a precise contract extraction engine. Output strict JSON only.",
        )

        data = parse_json_safely(response.text)

        return {
            "parties": BondParties(**(data.get("parties") or {})),
            "risks": RiskClauses(**(data.get("risks") or {})),
            "identified_citations": data.get("identified_citations") or [],
            "metrics_data": tracker.get_metrics(),
        }
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {"metrics_data": tracker.get_metrics()}


def _lookup_citation_worker(
    citation: str,
    model_name: str,
    use_google_search: bool,
) -> Tuple[str, StatuteRef, int, int]:
    """
    Worker that does ONE model call and returns:
      (citation, StatuteRef, input_tokens, output_tokens)
    """
    client = get_client()

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

    prompt = search_prompt if use_google_search else no_search_prompt

    resp = client.generate_content(
        prompt=prompt,
        model_name=model_name,
        response_mime_type="application/json",
        use_google_search=use_google_search,
        system_prompt="You are a legal research assistant. Output strict JSON only.",
    )

    data = parse_json_safely(resp.text)

    citation_data = {
        "citation": citation,
        "name": data.get("name", "Unknown"),
        "verbatim_text": data.get("verbatim_text"),
        "plain_summary": data.get("plain_summary"),
        "source_link": data.get("source_link"),
        "found": bool(data.get("found", False)),
    }

    if citation_data["found"] and not citation_data["source_link"]:
        citation_data["found"] = False
        citation_data["verbatim_text"] = "Statute identified but could not retrieve official text"
        citation_data["plain_summary"] = "Unable to verify statute details"

    return citation, StatuteRef(**citation_data), resp.input_tokens, resp.output_tokens


def research_node(state: BondState):
    print(f"--- BOND AGENT: RESEARCHING STATUTES ---")
    print(f"--- USING MODEL: {state.model_name} ---")

    tracker = MetricsTracker.from_dict(state.metrics_data, state.model_name)

    citations = state.identified_citations or []
    if not citations:
        return {"researched_statutes": [], "metrics_data": tracker.get_metrics()}

    config = SUPPORTED_MODELS.get(state.model_name)
    is_google_model = bool(config and config.provider == ModelProvider.GOOGLE)

    # Only Gemini can do grounded Google Search in this setup
    use_google_search = is_google_model

    # Parallelize lookups (bounded concurrency)
    max_workers = min(3, len(citations))
    results_by_citation: Dict[str, StatuteRef] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_lookup_citation_worker, c, state.model_name, use_google_search)
            for c in citations
        ]

        for fut in as_completed(futures):
            try:
                citation, statute_ref, in_tok, out_tok = fut.result()
                results_by_citation[citation] = statute_ref
                tracker.record_call(in_tok, out_tok)
            except Exception as e:
                # If one lookup fails, keep going
                msg = str(e)
                print(f"Error researching citation: {msg}")

    researched: List[StatuteRef] = []
    for c in citations:
        if c in results_by_citation:
            researched.append(results_by_citation[c])
        else:
            researched.append(
                StatuteRef(
                    citation=c,
                    name="Research Error",
                    verbatim_text="Unable to retrieve statute information",
                    plain_summary="Statute lookup failed",
                    source_link=None,
                    found=False,
                )
            )

    return {"researched_statutes": researched, "metrics_data": tracker.get_metrics()}


def opinion_node(state: BondState):
    print(f"--- BOND AGENT: SYNTHESIZING OPINION ---")
    print(f"--- USING MODEL: {state.model_name} ---")

    tracker = MetricsTracker.from_dict(state.metrics_data, state.model_name)
    client = get_client()

    parties_txt = state.parties.model_dump_json() if state.parties else ""
    risks_txt = state.risks.model_dump_json() if state.risks else ""
    statutes_txt = json.dumps([s.model_dump() for s in (state.researched_statutes or [])])

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
    """

    try:
        response = client.generate_content(
            prompt=prompt,
            model_name=state.model_name,
            response_mime_type="application/json",
            temperature=0.2,
            tracker=tracker,
            system_prompt="You are a conservative commercial surety underwriter. Output strict JSON only.",
        )

        data = parse_json_safely(response.text)
        opinion = UnderwritingOpinion(**data)

        final_payload = {
            "type": "BOND",
            "step_1": {
                "parties": state.parties.model_dump() if state.parties else {},
                "risks": state.risks.model_dump() if state.risks else {},
            },
            "step_2": {"statutes": [s.model_dump() for s in (state.researched_statutes or [])]},
            "step_3": {"opinion": opinion.model_dump()},
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
