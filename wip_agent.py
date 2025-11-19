import os
import json
from typing import List, Dict, Any, Optional
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

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-2.0-flash-exp"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.0
)

# ==========================================
# 2. DATA MODELS
# ==========================================

class FullWipRow(BaseModel):
    job_id: str = Field(description="Job Number or ID")
    job_name: Optional[str] = Field(default="", description="Job Name")
    
    total_contract_price: float = Field(default=0.0)
    estimated_total_costs: float = Field(default=0.0)
    estimated_gross_profit: float = Field(default=0.0)
    
    revenues_earned: float = Field(default=0.0)
    cost_to_date: float = Field(default=0.0)
    gross_profit_to_date: float = Field(default=0.0)
    
    billed_to_date: float = Field(default=0.0)
    cost_to_complete: float = Field(default=0.0)
    under_billings: float = Field(default=0.0)
    over_billings: float = Field(default=0.0)

class WipTotals(BaseModel):
    total_contract_price: float = 0.0
    estimated_total_costs: float = 0.0
    estimated_gross_profit: float = 0.0
    revenues_earned: float = 0.0
    cost_to_date: float = 0.0
    gross_profit_to_date: float = 0.0
    billed_to_date: float = 0.0
    cost_to_complete: float = 0.0
    under_billings: float = 0.0
    over_billings: float = 0.0

class CalculatedWipRow(FullWipRow):
    @computed_field
    @property
    def percent_complete(self) -> float:
        if self.estimated_total_costs and self.estimated_total_costs > 0:
            val = self.cost_to_date / self.estimated_total_costs
            return min(val, 1.0)
        return 0.0

class WipState(BaseModel):
    file_path: str
    processed_data: List[CalculatedWipRow] = []
    totals_row: Optional[WipTotals] = None
    final_json: Dict[str, Any] = {}

# ==========================================
# 3. EXTRACTOR NODE
# ==========================================

def extractor_node(state: WipState):
    print(f"\n--- EXTRACTING DATA FROM: {state.file_path} ---")
    
    try:
        with open(state.file_path, "rb") as f:
            file_bytes = f.read()
    except FileNotFoundError:
        return {"processed_data": [], "totals_row": None}

    prompt = """
    Extract the WIP Schedule table.
    1. Extract every job row with all financial columns, Job Name, and Job ID.
    2. Extract the "TOTALS" row from the bottom of the report.

    Return JSON:
    {
        "rows": [
            {
                "job_id": "string",
                "job_name": "string",
                "total_contract_price": number,
                "estimated_total_costs": number,
                "estimated_gross_profit": number,
                "revenues_earned": number,
                "cost_to_date": number,
                "gross_profit_to_date": number,
                "billed_to_date": number,
                "cost_to_complete": number,
                "under_billings": number,
                "over_billings": number
            }
        ],
        "totals": {
            "total_contract_price": number, 
            ... (same fields as rows)
        }
    }
    RULES:
    - (100) in parens is negative -100.
    - Empty fields are 0.
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        
        data = json.loads(response.text)
        rows = [CalculatedWipRow(**r) for r in data.get("rows", [])]
        
        totals = None
        if data.get("totals"):
            try:
                totals = WipTotals(**data["totals"])
            except: pass
        
        return {"processed_data": rows, "totals_row": totals}
        
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {"processed_data": [], "totals_row": None}

# ==========================================
# 4. ANALYST NODE (MATH & LOGIC)
# ==========================================

def analyst_node(state: WipState):
    print("--- RUNNING VALIDATIONS & ANALYSIS ---")
    rows = state.processed_data
    extracted_totals = state.totals_row
    
    if not rows:
        return {"final_json": {"error": "No data found"}}

    # --- 1. AGGREGATES ---
    calc = WipTotals()
    for r in rows:
        calc.total_contract_price += r.total_contract_price
        calc.estimated_total_costs += r.estimated_total_costs
        calc.estimated_gross_profit += r.estimated_gross_profit
        calc.revenues_earned += r.revenues_earned
        calc.cost_to_date += r.cost_to_date
        calc.gross_profit_to_date += r.gross_profit_to_date
        calc.billed_to_date += r.billed_to_date
        calc.cost_to_complete += r.cost_to_complete
        calc.under_billings += r.under_billings
        calc.over_billings += r.over_billings

    # --- 2. KPIs ---
    
    # UEGP = Unearned Gross Profit
    t_uegp = calc.estimated_gross_profit - calc.gross_profit_to_date
    
    # GP %
    gp_percent = (calc.gross_profit_to_date / calc.revenues_earned * 100) if calc.revenues_earned else 0
    
    # Net UB / OB: Sum(Abs(Over)) - Sum(Abs(Under))
    # We use abs() to be robust against OCR reading them as negatives
    total_over_abs = sum(abs(r.over_billings) for r in rows)
    total_under_abs = sum(abs(r.under_billings) for r in rows)
    net_ub_ob = total_over_abs - total_under_abs
    
    net_ub_ob_label = f"Over ${net_ub_ob/1000:.0f}k" if net_ub_ob >= 0 else f"Under ${abs(net_ub_ob)/1000:.0f}k"

    # --- 3. VALIDATIONS ---
    struct_pass = len(rows) > 0 and all(r.job_id for r in rows)
    
    # Formulaic check
    formula_failures = 0
    for r in rows:
        if abs((r.total_contract_price - r.estimated_total_costs) - r.estimated_gross_profit) > 1.0:
            formula_failures += 1
    formula_pass = formula_failures == 0

    # Totals check
    totals_pass = False
    totals_msg = "Fail"
    if extracted_totals:
        if abs(calc.revenues_earned - extracted_totals.revenues_earned) < 5.0:
            totals_pass = True
            totals_msg = "Pass"

    # --- 4. PORTFOLIO NARRATIVE ---
    loss_jobs = sum(1 for r in rows if r.estimated_gross_profit < 0)
    ub_jobs_count = sum(1 for r in rows if r.under_billings > r.over_billings)
    ub_pct = (ub_jobs_count / len(rows) * 100) if rows else 0
    
    if loss_jobs == 0:
        profit_text = "consistently profitable"
    else:
        profit_text = f"profitable, with {loss_jobs} jobs projecting losses"
        
    if ub_pct > 15:
        ub_text = f"Under billings seem to be a consistent issue with {ub_pct:.0f}% of jobs showing a negative position."
    else:
        ub_text = f"Billing cadence is healthy, with only {ub_pct:.0f}% of jobs currently underbilled."

    summary_text = (
        f"The portfolio is {profit_text}. {ub_text} "
        f"Overall Net UB / OB position is {net_ub_ob_label}."
    )

    # --- 5. RISK ANALYSIS ---
    risks = []
    sorted_rows = sorted(rows, key=lambda x: max(x.over_billings, x.under_billings), reverse=True)
    
    for row in sorted_rows[:5]:
        is_ob = row.over_billings > row.under_billings
        val = row.over_billings if is_ob else row.under_billings
        if val > 0:
            severity = "high" if val > 100000 else "medium"
            risks.append({
                "jobId": row.job_id,
                "jobName": row.job_name,
                "riskTags": "Overbilling" if is_ob else "Underbilling",
                "riskLevel": severity,
                "riskLevelLabel": severity.capitalize(),
                "amountAbs": f"${val:,.0f}",
                "ubobType": "OB" if is_ob else "UB",
                "percentComplete": f"{row.percent_complete:.1%}",
                "analysis": f"{'Billings exceed' if is_ob else 'Revenue exceeds'} progress by ${val:,.0f}."
            })

    payload = {
        "clean_table": [r.model_dump() for r in rows],
        "widget_data": {
            "summary": {"text": summary_text},
            "validations": {
                "structural": {"passed": struct_pass, "message": "Pass" if struct_pass else "Fail"},
                "formulaic": {"passed": formula_pass, "message": "Pass" if formula_pass else "Fail"},
                "totals": {"passed": totals_pass, "message": totals_msg}
            },
            "metrics": {
                "row1_1": {"label": "Contract Value", "value": f"${calc.total_contract_price/1000000:.2f}M"},
                "row1_2": {"label": "UEGP", "value": f"${t_uegp/1000000:.2f}M"},
                "row1_3": {"label": "CTC", "value": f"${calc.cost_to_complete/1000000:.2f}M"},
                "row2_1": {"label": "Earned Rev", "value": f"${calc.revenues_earned/1000000:.2f}M"},
                "row2_2": {"label": "GP %", "value": f"{gp_percent:.1f}%"},
                "row2_3": {"label": "Net UB / OB", "value": net_ub_ob_label}
            },
            "riskRowsAll": risks
        }
    }
    
    return {"final_json": payload}

workflow = StateGraph(WipState)
workflow.add_node("extract", extractor_node)
workflow.add_node("analyze", analyst_node)
workflow.set_entry_point("extract")
workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", END)
app = workflow.compile()