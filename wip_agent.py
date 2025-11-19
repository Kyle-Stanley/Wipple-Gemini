import os
import json
import math
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
    job_name: Optional[str] = Field(default="", description="Job Name or Description")
    
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
    """Adds calculated fields and normalization for the frontend"""
    
    @computed_field
    @property
    def percent_complete(self) -> float:
        if self.estimated_total_costs and self.estimated_total_costs > 0:
            val = self.cost_to_date / self.estimated_total_costs
            return min(val, 1.0) # Cap at 100% for display sanity, though >100 is possible
        return 0.0
    
    @computed_field
    @property
    def earned_revenue_calc(self) -> float:
        """Internal calc for validation comparison"""
        return self.total_contract_price * self.percent_complete

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
    Extract the WIP Schedule table. I need three specific things:
    1. Every single job row with all financial columns.
    2. The Job Name and Job ID for every row.
    3. The "TOTALS" row usually found at the bottom of the report.

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
    }

    RULES:
    - If a value is in parentheses (100), it is negative -100.
    - If a field is empty or dash, use 0.
    - Do not calculate values, extract exactly what is written.
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        
        data = json.loads(response.text)
        rows = [CalculatedWipRow(**r) for r in data.get("rows", [])]
        
        totals = None
        if data.get("totals"):
            # Ensure we handle empty dictionary or partial matches
            try:
                totals = WipTotals(**data["totals"])
            except:
                print("Warning: Totals row malformed")
        
        return {"processed_data": rows, "totals_row": totals}
        
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {"processed_data": [], "totals_row": None}

# ==========================================
# 4. ANALYST NODE (VALIDATION LOGIC)
# ==========================================

def analyst_node(state: WipState):
    print("--- RUNNING VALIDATIONS & ANALYSIS ---")
    rows = state.processed_data
    extracted_totals = state.totals_row
    
    if not rows:
        return {"final_json": {"error": "No data found"}}

    # --- 1. CALCULATE AGGREGATES ---
    calc_totals = WipTotals()
    for r in rows:
        calc_totals.total_contract_price += r.total_contract_price
        calc_totals.estimated_total_costs += r.estimated_total_costs
        calc_totals.estimated_gross_profit += r.estimated_gross_profit
        calc_totals.revenues_earned += r.revenues_earned
        calc_totals.cost_to_date += r.cost_to_date
        calc_totals.gross_profit_to_date += r.gross_profit_to_date
        calc_totals.billed_to_date += r.billed_to_date
        calc_totals.cost_to_complete += r.cost_to_complete
        calc_totals.under_billings += r.under_billings
        calc_totals.over_billings += r.over_billings

    # --- 2. PERFORM VALIDATIONS ---
    
    # A. Structural Validation (Did we get rows? Do they have IDs?)
    struct_pass = len(rows) > 0 and all(r.job_id for r in rows)
    struct_msg = "Structure Valid" if struct_pass else "Missing Job IDs or Empty Table"

    # B. Formulaic Validation (Row level logic)
    # Check: Contract - Est Cost = Est GP (Allowing $1.00 rounding diff)
    formula_failures = 0
    for r in rows:
        expected_gp = r.total_contract_price - r.estimated_total_costs
        if abs(expected_gp - r.estimated_gross_profit) > 1.0:
            formula_failures += 1
            
    formula_pass = formula_failures == 0
    formula_msg = "All formulas balance" if formula_pass else f"{formula_failures} rows have calculation errors"

    # C. Totals Validation (Sum of rows vs Extracted Total)
    totals_pass = False
    totals_msg = "No Totals Row Found"
    
    if extracted_totals:
        # Compare Revenue as a proxy for accuracy
        diff = abs(calc_totals.revenues_earned - extracted_totals.revenues_earned)
        if diff < 5.0: # Allow $5 rounding diff on grand total
            totals_pass = True
            totals_msg = "Sum of rows matches Report Total"
        else:
            totals_msg = f"Sum mismatch: Calc ${calc_totals.revenues_earned:,.0f} vs Rep ${extracted_totals.revenues_earned:,.0f}"

    # --- 3. RISK ANALYSIS ---
    risks = []
    # Sort by magnitude of billing variance
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
                "percentComplete": f"{row.percent_complete:.1%}"
            })

    # --- 4. PORTFOLIO OVERVIEW TEXT ---
    # Generate this deterministically to ensure it always appears
    gp_percent = (calc_totals.gross_profit_to_date / calc_totals.revenues_earned * 100) if calc_totals.revenues_earned else 0
    
    summary_text = (
        f"Portfolio contains {len(rows)} jobs with Total Contract Value of ${calc_totals.total_contract_price:,.0f}. "
        f"Cumulative Gross Profit is running at {gp_percent:.1f}%. "
        f"Identified {len(risks)} key billing variances requiring review."
    )

    # --- 5. CONSTRUCT FINAL PAYLOAD ---
    payload = {
        "clean_table": [r.model_dump() for r in rows],
        "widget_data": {
            "summary": {"text": summary_text},
            "validations": {
                "structural": {"passed": struct_pass, "message": struct_msg},
                "formulaic": {"passed": formula_pass, "message": formula_msg},
                "totals": {"passed": totals_pass, "message": totals_msg}
            },
            "metrics": {
                "total_contract": {"label": "Contract", "value": f"${calc_totals.total_contract_price/1000000:.2f}M"},
                "earned": {"label": "Earned Rev", "value": f"${calc_totals.revenues_earned/1000000:.2f}M"},
                "billed": {"label": "Billed", "value": f"${calc_totals.billed_to_date/1000000:.2f}M"},
                "gp": {"label": "Gross Profit", "value": f"${calc_totals.gross_profit_to_date/1000000:.2f}M"},
                "gp_pct": {"label": "GP Margin", "value": f"{gp_percent:.1f}%"},
                "net_bill": {"label": "Net Billing", "value": f"${(calc_totals.over_billings - calc_totals.under_billings)/1000:.0f}k"}
            },
            "riskRowsAll": risks
        }
    }
    
    return {"final_json": payload}

# ==========================================
# 5. WORKFLOW
# ==========================================

workflow = StateGraph(WipState)
workflow.add_node("extract", extractor_node)
workflow.add_node("analyze", analyst_node)
workflow.set_entry_point("extract")
workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", END)
app = workflow.compile()