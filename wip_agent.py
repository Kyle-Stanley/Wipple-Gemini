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

    # 1. Aggregates
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
        calc_totals = calc # Alias for easier reading below
        calc.over_billings += r.over_billings

    # 2. Specific KPIs requested
    # UEGP = Unearned Gross Profit = Estimated GP - Realized GP
    # Or: (Contract - Est Cost) - (Earned - CostToDate)
    t_uegp = calc.estimated_gross_profit - calc.gross_profit_to_date
    
    # GP %
    gp_percent = (calc.gross_profit_to_date / calc.revenues_earned * 100) if calc.revenues_earned else 0
    
    # Net Billings
    net_billings = calc.over_billings - calc.under_billings
    net_bill_label = f"Over ${net_billings/1000:.0f}k" if net_billings > 0 else f"Under ${abs(net_billings)/1000:.0f}k"

    # 3. Validations
    struct_pass = len(rows) > 0 and all(r.job_id for r in rows)
    
    # Formula check: Contract - Est Cost = Est GP
    formula_failures = 0
    for r in rows:
        if abs((r.total_contract_price - r.estimated_total_costs) - r.estimated_gross_profit) > 1.0:
            formula_failures += 1
    formula_pass = formula_failures == 0

    # Totals check
    totals_pass = False
    totals_msg = "No Totals Row"
    if extracted_totals:
        if abs(calc.revenues_earned - extracted_totals.revenues_earned) < 5.0:
            totals_pass = True
            totals_msg = "Matches Report Total"
        else:
            totals_msg = f"Calc ${calc.revenues_earned:,.0f} vs Rep ${extracted_totals.revenues_earned:,.0f}"

    # 4. Risk Analysis
    risks = []
    sorted_rows = sorted(rows, key=lambda x: max(x.over_billings, x.under_billings), reverse=True)
    
    for row in sorted_rows[:5]:
        is_ob = row.over_billings > row.under_billings
        val = row.over_billings if is_ob else row.under_billings
        if val > 0:
            severity = "high" if val > 100000 else "medium"
            # Generate analysis text
            if is_ob:
                analysis = f"Billings exceed completion by ${val:,.0f}. Verify cash flow advantage vs liability."
            else:
                analysis = f"Revenue recognition lags billing by ${val:,.0f}. Potential cash flow drag."
                
            risks.append({
                "jobId": row.job_id,
                "jobName": row.job_name,
                "riskTags": "Overbilling" if is_ob else "Underbilling",
                "riskLevel": severity,
                "riskLevelLabel": severity.capitalize(),
                "amountAbs": f"${val:,.0f}",
                "ubobType": "OB" if is_ob else "UB",
                "percentComplete": f"{row.percent_complete:.1%}",
                "analysis": analysis
            })

    summary_text = (
        f"Analyzed {len(rows)} jobs. Portfolio maintains {gp_percent:.1f}% margin with {net_bill_label} position. "
        f"Validation passed on {len(rows)-formula_failures}/{len(rows)} rows."
    )

    payload = {
        "clean_table": [r.model_dump() for r in rows],
        "widget_data": {
            "summary": {"text": summary_text},
            "validations": {
                "structural": {"passed": struct_pass, "message": "Structure Valid" if struct_pass else "Check Data"},
                "formulaic": {"passed": formula_pass, "message": "Formulas Balance" if formula_pass else "Calc Errors"},
                "totals": {"passed": totals_pass, "message": totals_msg}
            },
            # EXACT ORDER REQUESTED
            "metrics": {
                "row1_1": {"label": "Contract Value", "value": f"${calc.total_contract_price/1000000:.2f}M"},
                "row1_2": {"label": "UEGP", "value": f"${t_uegp/1000000:.2f}M"},
                "row1_3": {"label": "CTC", "value": f"${calc.cost_to_complete/1000000:.2f}M"},
                "row2_1": {"label": "Earned Rev", "value": f"${calc.revenues_earned/1000000:.2f}M"},
                "row2_2": {"label": "GP %", "value": f"{gp_percent:.1f}%"},
                "row2_3": {"label": "Net Billings", "value": net_bill_label}
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