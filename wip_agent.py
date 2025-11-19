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
        
    @computed_field
    @property
    def uegp(self) -> float:
        # Unearned Gross Profit = Est GP - GP to Date
        return self.estimated_gross_profit - self.gross_profit_to_date

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

    # --- 1. AGGREGATES & RECALCULATION ---
    calc = WipTotals()
    
    calculated_net_variance_sum = 0.0
    
    # Collect detailed errors for validation lists
    billing_logic_errors = [] # Stores {id, msg}
    basic_math_errors = []    # Stores {id, msg}
    
    for r in rows:
        # Standard sums
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
        
        # --- User's Logic for Net UB/OB ---
        poc = 0.0
        if r.estimated_total_costs and r.estimated_total_costs != 0:
            poc = r.cost_to_date / r.estimated_total_costs
        
        expected_revenue = poc * r.total_contract_price
        row_variance = r.billed_to_date - expected_revenue
        calculated_net_variance_sum += row_variance
        
        # --- VALIDATION A: Billing Logic ---
        # Reported Net = Over - Under
        reported_net = r.over_billings - r.under_billings
        if abs(reported_net - row_variance) > 100.0: 
             billing_logic_errors.append({
                 "id": r.job_id, 
                 "msg": f"Reported Net ${reported_net:,.0f} vs Calculated ${row_variance:,.0f}"
             })
             
        # --- VALIDATION B: Basic Math ---
        # Contract - Cost = GP
        if abs((r.total_contract_price - r.estimated_total_costs) - r.estimated_gross_profit) > 1.0:
            basic_math_errors.append({
                "id": r.job_id,
                "msg": f"Contract - Cost ({r.total_contract_price - r.estimated_total_costs:,.0f}) != GP ({r.estimated_gross_profit:,.0f})"
            })

    # --- 2. KPIs ---
    t_uegp = calc.estimated_gross_profit - calc.gross_profit_to_date
    gp_percent = (calc.gross_profit_to_date / calc.revenues_earned * 100) if calc.revenues_earned else 0
    net_ub_ob = calculated_net_variance_sum
    net_ub_ob_label = f"Over ${net_ub_ob/1000:.0f}k" if net_ub_ob >= 0 else f"Under ${abs(net_ub_ob)/1000:.0f}k"

    # --- 3. VALIDATIONS ---
    # A. Structural
    struct_pass = len(rows) > 0 and all(r.job_id for r in rows)
    struct_msg = "Structure Valid" if struct_pass else "Missing IDs/Data"
    
    # B. Formulaic
    total_failures = len(basic_math_errors) + len(billing_logic_errors)
    formula_pass = total_failures == 0
    
    if formula_pass:
        formula_msg = "Formulas & Billing Logic Balance"
    elif len(billing_logic_errors) > 0:
        formula_msg = f"Billing Logic Mismatch ({len(billing_logic_errors)} rows)"
    else:
        formula_msg = f"Basic Math Errors ({len(basic_math_errors)} rows)"
        
    # Combine errors for display
    all_formula_errors = basic_math_errors + billing_logic_errors

    # C. Totals
    totals_pass = False
    totals_msg = "No Totals Row"
    totals_details = []
    if extracted_totals:
        diff = abs(calc.revenues_earned - extracted_totals.revenues_earned)
        if diff < 5.0:
            totals_pass = True
            totals_msg = "Sum matches Report Total"
        else:
            totals_msg = f"Sum Mismatch (${diff:,.0f})"
            totals_details.append({"id": "TOTALS", "msg": f"Calc Earned ${calc.revenues_earned:,.0f} vs Report ${extracted_totals.revenues_earned:,.0f}"})

    # --- 4. PORTFOLIO NARRATIVE ---
    loss_jobs = sum(1 for r in rows if r.estimated_gross_profit < 0)
    ub_jobs_count = 0
    for r in rows:
        poc = (r.cost_to_date / r.estimated_total_costs) if r.estimated_total_costs else 0
        expected = poc * r.total_contract_price
        if r.billed_to_date < expected:
             ub_jobs_count += 1    
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
    rows_with_variance = []
    for r in rows:
        poc = (r.cost_to_date / r.estimated_total_costs) if r.estimated_total_costs else 0
        expected = poc * r.total_contract_price
        variance = r.billed_to_date - expected
        rows_with_variance.append((r, variance))
        
    sorted_rows = sorted(rows_with_variance, key=lambda x: abs(x[1]), reverse=True)
    
    for row, variance in sorted_rows[:5]:
        is_ob = variance > 0
        val = abs(variance)
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
                "structural": {"passed": struct_pass, "message": struct_msg, "details": []},
                "formulaic": {"passed": formula_pass, "message": formula_msg, "details": all_formula_errors},
                "totals": {"passed": totals_pass, "message": totals_msg, "details": totals_details}
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