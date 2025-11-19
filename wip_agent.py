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
    """Direct reconstruction of the WIP row exactly as reported."""
    job_id: str = Field(description="Job Number or ID as shown on the WIP")
    job_name: str | None = Field(default=None, description="Job description / project name")

    total_contract_price: float | None = Field(default=None, description="Revised/Total Contract Price")
    est_total_costs: float | None = Field(default=None, description="Estimated Total Costs at Completion")
    est_gross_profit: float | None = Field(default=None, description="Estimated Gross Profit")

    revenues_earned: float | None = Field(default=None, description="Revenues Earned / Earned Revenue to date")
    cost_to_date: float | None = Field(default=None, description="Job-to-Date Costs / Cost to Date")
    gross_profit_to_date: float | None = Field(default=None, description="Gross Profit to Date")

    billed_to_date: float | None = Field(default=None, description="Billings to Date / Total Billed")
    cost_to_complete: float | None = Field(default=None, description="Cost to Complete")

    under_billings: float | None = Field(default=None, description="Underbillings as reported")
    over_billings: float | None = Field(default=None, description="Overbillings as reported")


class WipTotals(BaseModel):
    """The TOTAL row at the bottom of the WIP."""
    total_contract_price: float | None = None
    est_total_costs: float | None = None
    est_gross_profit: float | None = None

    revenues_earned: float | None = None
    cost_to_date: float | None = None
    gross_profit_to_date: float | None = None

    billed_to_date: float | None = None
    cost_to_complete: float | None = None

    under_billings: float | None = None
    over_billings: float | None = None


class CalculatedWipRow(RawWipRow):
    """Python Logic Layer — never overwrites reported values."""

    @computed_field
    def percent_complete(self) -> float:
        if self.est_total_costs and self.est_total_costs != 0 and self.cost_to_date is not None:
            return round(self.cost_to_date / self.est_total_costs, 4)
        return 0.0

    @computed_field
    def earned_revenue_calc(self) -> float:
        """Earned revenue = contract * percent complete (fallback if missing)."""
        if self.total_contract_price is not None:
            return round(self.total_contract_price * self.percent_complete, 2)
        return 0.0

    @computed_field
    def over_billing_calc(self) -> float:
        """Overbilling when reported value missing."""
        if self.billed_to_date is None:
            return 0.0
        earned = self.revenues_earned if self.revenues_earned is not None else self.earned_revenue_calc
        val = self.billed_to_date - earned
        return round(val, 2) if val > 0 else 0.0

    @computed_field
    def under_billing_calc(self) -> float:
        """Underbilling when reported value missing."""
        if self.billed_to_date is None:
            return 0.0
        earned = self.revenues_earned if self.revenues_earned is not None else self.earned_revenue_calc
        val = earned - self.billed_to_date
        return round(val, 2) if val > 0 else 0.0

    @computed_field
    def ctc_calc(self) -> float:
        """Cost to complete (calc) = est_total_costs - cost_to_date."""
        if self.est_total_costs is not None and self.cost_to_date is not None:
            return round(self.est_total_costs - self.cost_to_date, 2)
        return 0.0


class WipExtractionResult(BaseModel):
    """Structured output returned by the extraction LLM."""
    rows: List[RawWipRow]
    totals: WipTotals


class WipState(BaseModel):
    """State object passed between LangGraph nodes."""
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

    prompt =     prompt = """
    You are extracting a Work-In-Progress (WIP) schedule from a construction contractor's financial statement.

    GOAL:
    - Recreate the WIP table EXACTLY as reported for the columns we care about.
    - One JSON row per job.
    - Also return a separate "totals" object that represents the TOTAL row at the bottom of the WIP.

    COLUMNS TO EXTRACT (FOR EACH JOB ROW):

    1) job_id
       - Map from columns like: "Job No", "Job #", "Job", "Contract #", "Project No".
       - Use the most specific job identifier shown.

    2) job_name
       - Map from job description / project description / job name.

    3) total_contract_price
       - Map from: "Total Contract", "Revised Contract", "Revised Contract Amount", "Contract Price".
       - Use the dollar value shown on the WIP (DO NOT recompute).

    4) est_total_costs
       - Map from: "Estimated Cost", "Est. Costs at Completion", "Estimated Total Costs", "Est Cost".
       - Use the dollar value shown on the WIP.

    5) est_gross_profit
       - Map from: "Estimated Gross Profit", "Gross Profit", "GP at Completion".
       - Use the value shown on the WIP. Do not recompute unless it is explicitly not present.
       - If not present anywhere, set to null.

    6) revenues_earned
       - Map from: "Earned Revenue", "Revenue Earned", "Costs and Estimated Earnings in Excess of Billings",
         or the standard WIP column that shows revenue recognized to date for the job.
       - Use the WIP number as printed. Do not recompute unless the column is clearly missing.

    7) cost_to_date
       - Map from: "Cost to Date", "Job-to-Date Costs", "JTD Cost", "Costs Incurred to Date".

    8) gross_profit_to_date
       - Map from: "Gross Profit to Date", "GP to Date".
       - If not present on the WIP, set to null (do NOT recompute here).

    9) billed_to_date
       - Map from: "Billed to Date", "Billings to Date", "Total Billings", "Progress Billings".

    10) cost_to_complete
        - Map from: "Cost to Complete", "CTC".
        - If not explicitly present, set to null (do NOT recompute here).

    11) under_billings
        - Map from: "Under Billings", "Underbillings".
        - Use the value as printed. Do not recompute.
        - If the WIP combines over/under into one column, interpret the sign:
          - Negative = under_billings
          - Positive = over_billings
          and split appropriately.

    12) over_billings
        - Map from: "Over Billings", "Overbillings".
        - Same rules as under_billings.

    RULES:
    - Ignore any "Total" or "Totals" row when building the rows list.
    - Numeric values must be plain numbers (no commas, no parentheses) and:
        - Convert "$(1,234)" or "(1,234)" to -1234.0
        - Convert "$1,234" to 1234.0
    - If a specific column does not exist in the WIP, set that field to null for that row.
      DO NOT invent values.
    - Preserve the sign as shown on the WIP.

    TOTALS ROW:
    - Also extract the TOTAL row at the bottom of the WIP, if present, and map it into the "totals" object.
    - For the totals:
        - total_contract_price, est_total_costs, est_gross_profit, revenues_earned,
          cost_to_date, gross_profit_to_date, billed_to_date, cost_to_complete,
          under_billings, over_billings
      should match the "Total" line for those columns (or null if not present).

    OUTPUT FORMAT:
    - Return a JSON object that matches the WipExtractionResult schema:
        {
          "rows": [...],
          "totals": { ... }
        }
    - "rows" is a list of RawWipRow objects (one per job).
    - "totals" is a single WipTotals object from the bottom "Total" row.
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
        extraction = WipExtractionResult(**data)

        raw_rows = extraction.rows
        calculated_rows = [CalculatedWipRow(**r.model_dump()) for r in raw_rows]

        print(f"--- EXTRACTED {len(calculated_rows)} ROWS ---")

        # You may want to keep totals in the state as well
        return {
            "processed_data": calculated_rows,
            "final_json": {
                "clean_table": [row.model_dump() for row in calculated_rows],
                "totals": extraction.totals.model_dump()
            }
        }

        
    except Exception as e:
        print(f"EXTRACTION ERROR: {e}")
        return {"processed_data": []}


def analyst_node(state: WipState):
    print("--- ANALYZING DATA ---")
    data = state.processed_data

    if not data:
        print("NO DATA. SKIPPING.")
        return {"final_json": {}}

    # Aggregations based on CalculatedWipRow
    t_contract = sum(r.total_contract_price or 0 for r in data)
    t_earned = sum((r.revenues_earned or r.earned_revenue_calc) for r in data)
    t_cost = sum((r.cost_to_date or 0) for r in data)
    gp_pct = ((t_earned - t_cost) / t_earned * 100) if t_earned else 0

    # Over/under using REPORTED values if present; fall back to calc
    def ub_ob(row: CalculatedWipRow):
        ub = row.under_billings
        ob = row.over_billings
        if ub is None and ob is None:
            return row.under_billing_calc, row.over_billing_calc
        return ub or 0.0, ob or 0.0

    risks = []
    with_variance = []
    for r in data:
        ub, ob = ub_ob(r)
        variance = max(ub, ob)
        with_variance.append((r, variance, ub, ob))

    sorted_jobs = sorted(with_variance, key=lambda x: x[1], reverse=True)

    for row, variance, ub, ob in sorted_jobs[:5]:
        severity = "high" if variance > 100000 else "medium"
        ubob_type = "OB" if ob > ub else "UB"
        risks.append({
            "id": row.job_id,
            "jobId": row.job_id,
            "riskTags": "Overbilling" if ubob_type == "OB" else "Underbilling",
            "riskLevel": severity,
            "riskLevelLabel": severity.capitalize(),
            "amountAbs": f"${variance:,.0f}",
            "ubobType": ubob_type,
            "percentComplete": f"{row.percent_complete:.1%}"
        })

    # Narrative
    response = llm.invoke(
        f"Write 1 concise, professional sentence summarizing the portfolio health. "
        f"Total Contract: ${t_contract:,.0f}. GP: {gp_pct:.1f}%. Risks: {len(risks)} jobs."
    ).content
    narrative = response.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    widget = {
        "summary": {"type": "text", "text": narrative},
        "metrics": {
            "total_contract": {"label": "Total Contract", "value": f"${t_contract/1_000_000:.2f}M"},
            "uegp": {"label": "UEGP", "value": "TBD"},  # plug your existing calc if you like
            "gp_percent": {"label": "GP%", "value": f"{gp_pct:.1f}%"},
            "ctc": {"label": "CTC", "value": "TBD"},
            "wip_gp_percent": {"label": "WIP GP%", "value": f"{gp_pct:.1f}%"},
            "cc_gp_percent": {"label": "CC GP%", "value": "N/A"}
        },
        "riskRowsAll": risks
    }

    payload = {
        "clean_table": [row.model_dump() for row in data],
        "totals": state.final_json.get("totals", {}),  # from extractor if you want it
        "widget_data": widget
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