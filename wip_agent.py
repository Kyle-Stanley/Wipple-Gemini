import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field, computed_field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types

# ==========================================
# 1. CONFIGURATION
# ==========================================

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "" 

# Initialize the NEW 2025 Client for native PDF reading
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-3-pro-preview"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.0
)

# ==========================================
# 2. THE DATA MODELS
# ==========================================

class RawWipRow(BaseModel):
    job_id: str = Field(description="Job Number or ID")
    contract_amount: float = Field(description="Revised Contract Value")
    est_cost: float = Field(description="Estimated Cost at Completion")
    billed_to_date: float = Field(description="Total Billed")
    cost_to_date: float = Field(description="Total Cost Incurred")

class CalculatedWipRow(RawWipRow):
    """The Python Logic Engine."""
    @computed_field
    def percent_complete(self) -> float:
        return round(self.cost_to_date / self.est_cost, 4) if self.est_cost else 0.0

    @computed_field
    def earned_revenue(self) -> float:
        return round(self.contract_amount * self.percent_complete, 2)

    @computed_field
    def over_billing(self) -> float:
        val = self.billed_to_date - self.earned_revenue
        return round(val, 2) if val > 0 else 0.0

    @computed_field
    def under_billing(self) -> float:
        val = self.earned_revenue - self.billed_to_date
        return round(val, 2) if val > 0 else 0.0
    
    @computed_field
    def ctc(self) -> float:
        return round(self.est_cost - self.cost_to_date, 2)

class WipExtractionResult(BaseModel):
    rows: List[RawWipRow]

class WipState(BaseModel):
    file_path: str
    processed_data: List[CalculatedWipRow] = []
    final_json: Dict[str, Any] = {}

# ==========================================
# 3. THE AGENTS (Hybrid Implementation)
# ==========================================

def extractor_node(state: WipState):
    print(f"\n--- GEMINI 3 PRO IS READING (NATIVE SDK): {state.file_path} ---")
    
    try:
        with open(state.file_path, "rb") as f:
            file_bytes = f.read()
    except FileNotFoundError:
        return {"processed_data": []}

    # Setup NATIVE Model for structured output
    native_model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": WipExtractionResult
        }
    )

    prompt = """
Extract the WIP Schedule table from the PDF.

You MUST return every job row and the TOTAL row exactly as they appear,
mapping them into the following fields:

- job_id
- job_name
- total_contract_price
- estimated_total_costs
- estimated_gross_profit
- revenues_earned
- cost_to_date
- gross_profit_to_date
- billed_to_date
- cost_to_complete
- under_billings
- over_billings

MAPPING RULES:
- Map “Job”, “Job #”, “Project No” → job_id
- Map Job Description / Project Name → job_name
- Map “Revised Contract”, “Contract Price”, “Total Contract” → total_contract_price
- Map “Est. Cost”, “Estimated Costs at Completion”, “Est Total Costs” → estimated_total_costs
- Map “Gross Profit”, “Est GP” → estimated_gross_profit
- Map “Earned Revenue”, “Revenues Earned”, “Earned to Date” → revenues_earned
- Map “Cost to Date”, “JTD Cost” → cost_to_date
- Map “Gross Profit To Date”, “GP to Date” → gross_profit_to_date
- Map “Billings to Date”, “Billed to Date”, “Total Billings” → billed_to_date
- Map “Cost to Complete”, “CTC” → cost_to_complete
- Map “Under Billings” → under_billings
- Map “Over Billings” → over_billings

ADDITIONAL RULES:
- Return the TOTAL row in a separate object called "totals".
- Ignore any rows labeled “Totals” when building job rows.
- Convert currency strings (“(1,234)”, “$1,234.00”) into plain numbers.
- If a column does not exist in the PDF, return null for that field.

OUTPUT FORMAT:
Produce a JSON object with:
{
  "rows": [...],
  "totals": { ... }
}
"""

    try:
        # Pass the PDF directly to Gemini 3 as bytes
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WipExtractionResult
            )
        )
        
        # Parse JSON from response
        data = json.loads(response.text)
        raw_rows = [RawWipRow(**row) for row in data["rows"]]
        
        # Trigger Python Math Engine
        calculated_rows = [CalculatedWipRow(**r.model_dump()) for r in raw_rows]
        
        print(f"--- EXTRACTED {len(calculated_rows)} ROWS ---")
        return {"processed_data": calculated_rows}
        
    except Exception as e:
        print(f"EXTRACTION ERROR: {e}")
        return {"processed_data": []}


def analyst_node(state: WipState):
    print("--- ANALYZING DATA ---")
    data = state.processed_data
    
    if not data:
        print("NO DATA. SKIPPING.")
        return {"final_json": {}}

    # Aggregations
    t_contract = sum(r.contract_amount for r in data)
    t_uegp = sum((r.contract_amount - r.est_cost) - (r.earned_revenue - r.cost_to_date) for r in data)
    t_earned = sum(r.earned_revenue for r in data)
    t_cost = sum(r.cost_to_date for r in data)
    gp_pct = ((t_earned - t_cost) / t_earned * 100) if t_earned else 0
    
    risks = []
    sorted_jobs = sorted(data, key=lambda x: max(x.over_billing, x.under_billing), reverse=True)
    
    for job in sorted_jobs[:5]: 
        variance = max(job.over_billing, job.under_billing)
        severity = "high" if variance > 100000 else "medium"
        risks.append({
            "id": job.job_id,
            "jobId": job.job_id,
            "riskTags": "Overbilling" if job.over_billing > 0 else "Underbilling",
            "riskLevel": severity,
            "riskLevelLabel": severity.capitalize(),
            "amountAbs": f"${variance:,.0f}",
            "ubobType": "OB" if job.over_billing > 0 else "UB",
            "percentComplete": f"{job.percent_complete:.1%}"
        })

    # Narrative Generation
    # This LLM call produces a complex object with the text and metadata.
    # We must access the .content property to get the string, and then clean it.
    response = llm.invoke(f"Write 1 concise, professional sentence summarizing the portfolio health. Total Contract: ${t_contract}. GP: {gp_pct:.1f}%. Risks: {len(risks)} jobs.").content
    
    narrative = response.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    payload = {
        "clean_table": [row.model_dump() for row in data],
        "widget_data": {
            # We wrap the cleaned string in the desired object structure for the frontend
            "summary": {"type": "text", "text": narrative},
            "metrics": {
                "totalContractAmount": {"label": "Total Contract", "value": f"${t_contract/1000000:.2f}M"},
                "uegp": {"label": "UEGP", "value": f"${t_uegp/1000000:.2f}M"},
                "gpPct": {"label": "GP%", "value": f"{gp_pct:.1f}%"},
                "ctc": {"label": "CTC", "value": f"${sum(r.ctc for r in data)/1000000:.2f}M"},
                "wipGpPct": {"label": "WIP GP%", "value": f"{gp_pct:.1f}%"},
                "ccGpPct": {"label": "CC GP%", "value": "N/A"}
            },
            "riskRowsAll": risks
        }
    }
    return {"final_json": payload}

# ==========================================
# 4. EXECUTION FLOW
# ==========================================

workflow = StateGraph(WipState)
workflow.add_node("extract", extractor_node)
workflow.add_node("analyze", analyst_node)
workflow.set_entry_point("extract")
workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", END)
app = workflow.compile()

if __name__ == "__main__":
    TEST_FILE = "test_wip.pdf" 
    if os.path.exists(TEST_FILE):
        result = app.invoke({"file_path": TEST_FILE})
        if result.get("final_json"):
            print("\n--- FINAL JSON OUTPUT ---")
            print(json.dumps(result["final_json"]["widget_data"], indent=2))
    else:
        print(f"ERROR: Please put a file named '{TEST_FILE}' in this folder.")