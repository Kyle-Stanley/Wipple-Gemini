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
MODEL_NAME = "gemini-3-pro-preview"

# FIX: Removed temperature=0.0 to prevent Gemini 3 looping issues
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME
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
    
    # Add computed fields for the totals row export
    uegp: float = 0.0 

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

    CRITICAL: UNDER vs OVER BILLINGS
    - UNDER BILLINGS (UB) happens when: Cost to Date > Billed to Date
      Column headers: "Costs in Excess of Billings" OR "Billings in Excess of Revenues" OR similar
      Formula: Cost to Date - Billed to Date (when positive)
      
    - OVER BILLINGS (OB) happens when: Billed to Date > Cost to Date  
      Column headers: "In Excess of Billings" OR "Earnings in Excess of Costs" OR similar
      Formula: Billed to Date - Cost to Date (when positive)

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
    - Match column headers to the definitions above carefully
    """

    try:
        # FIX: Removed temperature=0.0 to prevent looping
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
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
    
    # Error Lists
    billing_logic_errors = [] # For UB/OB math mismatches
    basic_math_errors = []    # For Contract/Cost/GP math mismatches
    
    # STRICT ACCOUNTING RECALCULATION (UB/OB)
    # We overwrite the extracted UB/OB with calculated versions to ensure the "validated" table is mathematically perfect
    for r in rows:
        # Calculate Variance
        variance = r.revenues_earned - r.billed_to_date
        
        if variance > 0:
            r.under_billings = variance
            r.over_billings = 0.0
        else:
            r.under_billings = 0.0
            r.over_billings = abs(variance)

    # MAIN LOOP: AGGREGATION & VALIDATION
    for r in rows:
        # A. Aggregate
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
        calc.uegp += r.uegp 
        
        # B. Calculate Core Metrics from scratch (The "Truth" Source)
        # 1. Percent Complete
        calc_poc = 0.0
        if r.estimated_total_costs and r.estimated_total_costs != 0:
            calc_poc = r.cost_to_date / r.estimated_total_costs
        
        # 2. Expected Earned Revenue
        calc_earned_rev = calc_poc * r.total_contract_price
        
        # 3. Expected Variance (UB/OB) position
        calc_variance = calc_earned_rev - r.billed_to_date
        calc_ub = max(0, calc_variance)
        calc_ob = max(0, -calc_variance)
        
        # 4. Cost to Complete
        calc_ctc = r.estimated_total_costs - r.cost_to_date
        
        # --- VALIDATION 1: Column Math Integrity ---
        # Check 1: Does Revenue match POC * Contract?
        # (Allowing tolerance for manual override or rounding)
        rev_diff = abs(r.revenues_earned - calc_earned_rev)
        if rev_diff > 5000 and r.revenues_earned > 0: # $5k tolerance for stored materials/rounding
             basic_math_errors.append({
                 "id": r.job_id, 
                 "msg": f"Rev Mismatch: Rpt ${r.revenues_earned:,.0f} vs Calc ${calc_earned_rev:,.0f}"
             })

        # Check 2: Does CTC match Est Cost - Cost to Date?
        ctc_diff = abs(r.cost_to_complete - calc_ctc)
        if ctc_diff > 100: 
             basic_math_errors.append({
                 "id": r.job_id,
                 "msg": f"CTC Math Error (Diff ${ctc_diff:,.0f})"
             })
             
        # Check 3: Basic Contract Math (Contract - Est Cost = Est GP)
        gp_diff = abs((r.total_contract_price - r.estimated_total_costs) - r.estimated_gross_profit)
        if gp_diff > 100:
            basic_math_errors.append({
                "id": r.job_id,
                "msg": f"Est GP Math Error (Diff ${gp_diff:,.0f})"
            })

    # --- 2. KPIs ---
    t_uegp = calc.estimated_gross_profit - calc.gross_profit_to_date
    gp_percent = (calc.gross_profit_to_date / calc.revenues_earned * 100) if calc.revenues_earned else 0
    
    net_ub_ob = calc.under_billings - calc.over_billings
    net_ub_ob_label = f"Under ${net_ub_ob/1000:.0f}k" if net_ub_ob >= 0 else f"Over ${abs(net_ub_ob)/1000:.0f}k"

    # --- 3. VALIDATION ROLLUP ---
    struct_pass = len(rows) > 0 and all(r.job_id for r in rows)
    struct_msg = "Structure Valid" if struct_pass else "Missing IDs/Data"
    
    total_failures = len(basic_math_errors)
    formula_pass = total_failures == 0
    
    if formula_pass:
        formula_msg = "Column Math Validated"
    else:
        formula_msg = f"Column Math Issues ({len(basic_math_errors)} rows)"

    # Totals Validation
    totals_pass = False
    totals_msg = "No Totals Row"
    totals_details = []
    if extracted_totals:
        diff = abs(calc.revenues_earned - extracted_totals.revenues_earned)
        if diff < 1000.0: # $1k tolerance for rounding
            totals_pass = True
            totals_msg = "Sum matches Report Total"
        else:
            totals_msg = f"Sum Mismatch (${diff:,.0f})"
            totals_details.append({"id": "TOTALS", "msg": f"Calc Earned ${calc.revenues_earned:,.0f} vs Report ${extracted_totals.revenues_earned:,.0f}"})

    # --- 4. RISK ANALYSIS (Cone of Silence) ---
    risks = []
    
    for r in rows:
        # Re-calculate these locally for risk logic
        poc = (r.cost_to_date / r.estimated_total_costs) if r.estimated_total_costs else 0
        expected_revenue = poc * r.total_contract_price
        
        # Actual variance: What is actually happening (Billed - Revenue)
        # Positive = Billed More (OB), Negative = Billed Less (UB)
        variance_actual = r.billed_to_date - expected_revenue
        
        # ALLOWABLE TOLERANCE (The Cone)
        # Tolerance tightens as job gets closer to 100%
        # At 0% complete, high tolerance. At 100% complete, near zero tolerance.
        # Base tolerance: 10% of contract value, scaled down by POC.
        # Plus a floor of $5,000 to avoid flagging small jobs.
        
        cone_width_factor = max(0.02, 0.10 * (1.0 - min(poc, 1.0))) # 10% tapering to 2%
        allowable_variance = (r.total_contract_price * cone_width_factor) + 5000
        
        diff_from_tolerance = abs(variance_actual) - allowable_variance
        
        if diff_from_tolerance > 0:
            # It is a risk
            is_ob = variance_actual > 0
            severity = "high" if diff_from_tolerance > 50000 else "medium"
            
            risk_type = "Aggressive Billing" if is_ob else "Underbilling Lag"
            
            analysis_text = (
                f"Billings are {'over' if is_ob else 'under'} revenue by ${abs(variance_actual):,.0f}, "
                f"exceeding the allowable risk tolerance (based on {poc*100:.0f}% completion) by ${diff_from_tolerance:,.0f}."
            )
            
            risks.append({
                "jobId": r.job_id,
                "jobName": r.job_name,
                "riskTags": risk_type,
                "riskLevel": severity,
                "riskLevelLabel": severity.capitalize(),
                "amountAbs": f"${abs(variance_actual):,.0f}", # Show the full variance amount
                "ubobType": "OB" if is_ob else "UB",
                "percentComplete": f"{poc:.1%}",
                "analysis": analysis_text
            })

    # Sort risks by severity (amount exceeding tolerance)
    risks.sort(key=lambda x: float(x['amountAbs'].replace('$','').replace(',','')), reverse=True)
    top_risks = risks[:5]

    # --- 5. PORTFOLIO NARRATIVE ---
    # More professional summary
    
    total_jobs = len(rows)
    risky_job_count = len(risks)
    portfolio_poc = (calc.cost_to_date / calc.estimated_total_costs) if calc.estimated_total_costs else 0
    
    summary_text = (
        f"This portfolio contains {total_jobs} active jobs with a Total Contract Value of ${calc.total_contract_price/1000000:.1f}M. "
        f"Aggregate progress is {portfolio_poc:.1%} complete. "
        f"The overall financial position is Net {net_ub_ob_label}. "
        f"However, {risky_job_count} jobs have been flagged for billing variances that exceed standard risk tolerances."
    )

    payload = {
        "clean_table": [r.model_dump() for r in rows],
        "calculated_totals": calc.model_dump(), 
        "widget_data": {
            "summary": {"text": summary_text},
            "validations": {
                "structural": {"passed": struct_pass, "message": struct_msg, "details": []},
                "formulaic": {"passed": formula_pass, "message": formula_msg, "details": basic_math_errors},
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
            "riskRowsAll": top_risks
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
