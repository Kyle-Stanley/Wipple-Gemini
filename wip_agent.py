import os
import json
import traceback
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field, computed_field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types

# ==========================================
# 1. CONFIGURATION
# ==========================================
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-3-pro-preview"
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
        return self.estimated_gross_profit - self.gross_profit_to_date

class WipState(BaseModel):
    file_path: str
    processed_data: List[CalculatedWipRow] = []
    totals_row: Optional[WipTotals] = None
    calculated_totals: Optional[WipTotals] = None
    validation_errors: List[Dict[str, Any]] = []
    correction_suggestions: List[Dict[str, Any]] = []
    surety_risk_context: Dict[str, Any] = {}
    risk_rows: List[Dict[str, Any]] = []
    widget_data: Dict[str, Any] = {}
    narrative: str = ""
    final_json: Dict[str, Any] = {}

# ==========================================
# 3. VALIDATION REGISTRY
# ==========================================

@dataclass
class Validation:
    name: str
    requires: List[str]
    check: Callable[[CalculatedWipRow], Optional[str]]
    tolerance_type: str  # "absolute" or "percentage"
    tolerance_value: float
    category: str  # "core_math", "billing", "profitability", "completion"
    fields_involved: List[str]  # Fields that could be the source of error

def _check_billing_position(r: CalculatedWipRow) -> Optional[str]:
    """Validates that UB/OB values match the earned revenue vs billed calculation."""
    variance = r.revenues_earned - r.billed_to_date
    
    if variance > 0:
        expected_ub, expected_ob = variance, 0
    else:
        expected_ub, expected_ob = 0, abs(variance)
    
    ub_diff = abs(r.under_billings - expected_ub)
    ob_diff = abs(r.over_billings - expected_ob)
    
    if ub_diff > 100 or ob_diff > 100:
        return f"UB/OB doesn't match variance (expected UB ${expected_ub:,.0f}, OB ${expected_ob:,.0f})"
    return None

def build_validations() -> List[Validation]:
    return [
        # === CORE MATH (these should always hold) ===
        Validation(
            name="contract_cost_gp",
            requires=["total_contract_price", "estimated_total_costs", "estimated_gross_profit"],
            check=lambda r: (
                f"Contract - Est Cost ≠ Est GP (diff ${abs((r.total_contract_price - r.estimated_total_costs) - r.estimated_gross_profit):,.0f})"
                if abs((r.total_contract_price - r.estimated_total_costs) - r.estimated_gross_profit) > 100
                else None
            ),
            tolerance_type="absolute",
            tolerance_value=100,
            category="core_math",
            fields_involved=["total_contract_price", "estimated_total_costs", "estimated_gross_profit"]
        ),
        Validation(
            name="cost_to_complete_check",
            requires=["estimated_total_costs", "cost_to_date", "cost_to_complete"],
            check=lambda r: (
                f"Est Cost - Cost to Date ≠ CTC (diff ${abs((r.estimated_total_costs - r.cost_to_date) - r.cost_to_complete):,.0f})"
                if abs((r.estimated_total_costs - r.cost_to_date) - r.cost_to_complete) > 100
                else None
            ),
            tolerance_type="absolute",
            tolerance_value=100,
            category="core_math",
            fields_involved=["estimated_total_costs", "cost_to_date", "cost_to_complete"]
        ),
        Validation(
            name="earned_revenue_from_gp",
            requires=["revenues_earned", "cost_to_date", "gross_profit_to_date"],
            check=lambda r: (
                f"Cost + GP to Date ≠ Earned Rev (diff ${abs(r.revenues_earned - (r.cost_to_date + r.gross_profit_to_date)):,.0f})"
                if abs(r.revenues_earned - (r.cost_to_date + r.gross_profit_to_date)) > 100
                else None
            ),
            tolerance_type="absolute",
            tolerance_value=100,
            category="core_math",
            fields_involved=["revenues_earned", "cost_to_date", "gross_profit_to_date"]
        ),
        Validation(
            name="remaining_gp_check",
            requires=["estimated_gross_profit", "gross_profit_to_date"],
            check=lambda r: (
                f"UEGP calculation mismatch"
                if r.estimated_gross_profit > 0 and r.gross_profit_to_date > r.estimated_gross_profit * 1.05
                else None
            ),
            tolerance_type="percentage",
            tolerance_value=0.05,
            category="core_math",
            fields_involved=["estimated_gross_profit", "gross_profit_to_date"]
        ),
        
        # === COMPLETION-BASED (POC method) ===
        Validation(
            name="earned_revenue_from_poc",
            requires=["revenues_earned", "total_contract_price", "cost_to_date", "estimated_total_costs"],
            check=lambda r: (
                f"Earned Rev ≠ Contract × POC (diff ${abs(r.revenues_earned - (r.total_contract_price * (r.cost_to_date / r.estimated_total_costs if r.estimated_total_costs else 0))):,.0f})"
                if r.estimated_total_costs and abs(r.revenues_earned - (r.total_contract_price * (r.cost_to_date / r.estimated_total_costs))) > max(5000, r.total_contract_price * 0.02)
                else None
            ),
            tolerance_type="percentage",
            tolerance_value=0.02,
            category="completion",
            fields_involved=["revenues_earned", "total_contract_price", "cost_to_date", "estimated_total_costs"]
        ),
        
        # === BILLING ===
        Validation(
            name="underbilling_overbilling",
            requires=["revenues_earned", "billed_to_date", "under_billings", "over_billings"],
            check=lambda r: _check_billing_position(r),
            tolerance_type="absolute",
            tolerance_value=100,
            category="billing",
            fields_involved=["revenues_earned", "billed_to_date", "under_billings", "over_billings"]
        ),
        
        # === PROFITABILITY ===
        Validation(
            name="gp_percentage_bounds",
            requires=["estimated_gross_profit", "total_contract_price"],
            check=lambda r: (
                f"Unusually high GP% ({(r.estimated_gross_profit / r.total_contract_price * 100):.1f}%)"
                if r.total_contract_price > 0 and r.estimated_gross_profit / r.total_contract_price > 0.50
                else None
            ),
            tolerance_type="percentage",
            tolerance_value=0.50,
            category="profitability",
            fields_involved=["estimated_gross_profit", "total_contract_price"]
        ),
    ]

def run_validations(row: CalculatedWipRow, validations: List[Validation]) -> List[Dict[str, Any]]:
    """Run all applicable validations for a row."""
    errors = []
    row_dict = row.model_dump()
    
    for v in validations:
        required_present = all(
            row_dict.get(field) is not None
            for field in v.requires
        )
        
        has_nonzero = any(
            row_dict.get(field, 0) != 0
            for field in v.requires
        )
        
        if not required_present or not has_nonzero:
            continue
            
        error_msg = v.check(row)
        if error_msg:
            errors.append({
                "job_id": row.job_id,
                "validation": v.name,
                "category": v.category,
                "message": error_msg,
                "fields_involved": v.fields_involved
            })
    
    return errors

# ==========================================
# 4. CORRECTION SUGGESTIONS
# ==========================================

@dataclass
class CorrectionSuggestion:
    job_id: str
    field: str
    current_value: float
    suggested_value: float
    confidence: str
    reasoning: str

def is_plausible_ocr_error(extracted: float, expected: float) -> bool:
    """Check if the difference looks like a typical OCR mistake."""
    if extracted == 0 or expected == 0:
        return False
        
    diff = abs(extracted - expected)
    
    # Single digit off by power of 10
    powers_of_10 = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    for p in powers_of_10:
        if abs(diff - p) < 1:
            return True
        # Also check multiples (e.g., off by 2000000)
        for mult in range(1, 10):
            if abs(diff - (p * mult)) < 1:
                return True
    
    # Check for transposed or single-digit differences
    ext_str = str(int(abs(extracted)))
    exp_str = str(int(abs(expected)))
    if len(ext_str) == len(exp_str):
        diffs = sum(1 for a, b in zip(ext_str, exp_str) if a != b)
        if diffs <= 2:
            return True
    
    # Missing or extra zero
    if abs(extracted * 10 - expected) < 1 or abs(extracted - expected * 10) < 1:
        return True
        
    return False

def suggest_corrections(row: CalculatedWipRow, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate correction suggestions based on validation errors."""
    suggestions = []
    
    for error in errors:
        validation_name = error["validation"]
        
        if validation_name == "contract_cost_gp":
            # Contract - Est Cost should = Est GP
            expected_gp = row.total_contract_price - row.estimated_total_costs
            if is_plausible_ocr_error(row.estimated_gross_profit, expected_gp):
                suggestions.append({
                    "job_id": row.job_id,
                    "field": "estimated_gross_profit",
                    "current_value": row.estimated_gross_profit,
                    "suggested_value": expected_gp,
                    "confidence": "high",
                    "reasoning": f"If Est GP were ${expected_gp:,.0f}, the formula Contract - Est Cost = Est GP would hold. Difference pattern suggests OCR error."
                })
            
            # Also check if est_costs might be wrong
            expected_costs = row.total_contract_price - row.estimated_gross_profit
            if is_plausible_ocr_error(row.estimated_total_costs, expected_costs):
                suggestions.append({
                    "job_id": row.job_id,
                    "field": "estimated_total_costs",
                    "current_value": row.estimated_total_costs,
                    "suggested_value": expected_costs,
                    "confidence": "medium",
                    "reasoning": f"If Est Costs were ${expected_costs:,.0f}, the formula would hold. Alternative to GP correction."
                })
                
        elif validation_name == "cost_to_complete_check":
            # Est Cost - Cost to Date should = CTC
            expected_ctc = row.estimated_total_costs - row.cost_to_date
            if is_plausible_ocr_error(row.cost_to_complete, expected_ctc):
                suggestions.append({
                    "job_id": row.job_id,
                    "field": "cost_to_complete",
                    "current_value": row.cost_to_complete,
                    "suggested_value": expected_ctc,
                    "confidence": "high",
                    "reasoning": f"If CTC were ${expected_ctc:,.0f}, the formula Est Cost - Cost to Date = CTC would hold."
                })
                
        elif validation_name == "earned_revenue_from_gp":
            # Cost to Date + GP to Date should = Revenues Earned
            expected_rev = row.cost_to_date + row.gross_profit_to_date
            if is_plausible_ocr_error(row.revenues_earned, expected_rev):
                suggestions.append({
                    "job_id": row.job_id,
                    "field": "revenues_earned",
                    "current_value": row.revenues_earned,
                    "suggested_value": expected_rev,
                    "confidence": "high",
                    "reasoning": f"If Earned Rev were ${expected_rev:,.0f}, the formula Cost + GP to Date = Earned Rev would hold."
                })
            
            # Check if GP to date might be wrong
            expected_gp_to_date = row.revenues_earned - row.cost_to_date
            if is_plausible_ocr_error(row.gross_profit_to_date, expected_gp_to_date):
                suggestions.append({
                    "job_id": row.job_id,
                    "field": "gross_profit_to_date",
                    "current_value": row.gross_profit_to_date,
                    "suggested_value": expected_gp_to_date,
                    "confidence": "medium",
                    "reasoning": f"If GP to Date were ${expected_gp_to_date:,.0f}, the formula would hold."
                })
    
    return suggestions

# ==========================================
# 5. SURETY RISK ANALYSIS
# ==========================================

def build_surety_risk_context(rows: List[CalculatedWipRow], calc: WipTotals) -> Dict[str, Any]:
    """Build a structured risk context focused on surety concerns."""
    
    # Loss jobs (negative GP)
    loss_jobs = [r for r in rows if r.estimated_gross_profit < 0]
    total_loss_exposure = sum(abs(r.estimated_gross_profit) for r in loss_jobs)
    
    # Severe underbilling: high % OR large absolute amount
    severe_ub_jobs = [
        r for r in rows 
        if (r.under_billings > r.total_contract_price * 0.10 and r.under_billings > 25000) or
           (r.under_billings > 100000)
    ]
    total_ub_exposure = sum(r.under_billings for r in severe_ub_jobs)
    
    # Jobs with GP fade (GP to date lagging expected GP at this completion %)
    fade_jobs = []
    for r in rows:
        if r.estimated_total_costs > 0 and r.estimated_gross_profit > 0:
            poc = r.cost_to_date / r.estimated_total_costs
            expected_gp_to_date = r.estimated_gross_profit * poc
            if expected_gp_to_date > 1000 and r.gross_profit_to_date < expected_gp_to_date * 0.7:
                fade_jobs.append({
                    "job_id": r.job_id,
                    "job_name": r.job_name,
                    "expected_gp": expected_gp_to_date,
                    "actual_gp": r.gross_profit_to_date,
                    "fade_pct": 1 - (r.gross_profit_to_date / expected_gp_to_date) if expected_gp_to_date else 0,
                    "contract": r.total_contract_price
                })
    
    # Concentration: any single job > 25% of portfolio
    concentration_jobs = [
        r for r in rows 
        if calc.total_contract_price > 0 and r.total_contract_price > calc.total_contract_price * 0.25
    ]
    
    # Remaining margin quality
    total_uegp = calc.estimated_gross_profit - calc.gross_profit_to_date
    thin_margin_jobs = [r for r in rows if r.total_contract_price > 0 and r.estimated_gross_profit < r.total_contract_price * 0.05]
    uegp_at_risk = sum(
        r.estimated_gross_profit - r.gross_profit_to_date 
        for r in thin_margin_jobs
        if r.estimated_gross_profit > r.gross_profit_to_date
    )
    
    # Overbilling (less critical but worth noting)
    severe_ob_jobs = [
        r for r in rows
        if (r.over_billings > r.total_contract_price * 0.15 and r.over_billings > 25000) or
           (r.over_billings > 100000)
    ]
    
    return {
        "portfolio": {
            "total_jobs": len(rows),
            "total_contract_value": calc.total_contract_price,
            "aggregate_poc": calc.cost_to_date / calc.estimated_total_costs if calc.estimated_total_costs else 0,
            "net_billing_position": calc.under_billings - calc.over_billings,
            "total_uegp": total_uegp,
            "total_gp_margin": calc.estimated_gross_profit / calc.total_contract_price if calc.total_contract_price else 0
        },
        "cash_risk": {
            "severe_ub_count": len(severe_ub_jobs),
            "severe_ub_jobs": [
                {"id": r.job_id, "name": r.job_name, "ub_amount": r.under_billings, "contract": r.total_contract_price} 
                for r in severe_ub_jobs
            ],
            "total_ub_exposure": total_ub_exposure
        },
        "loss_risk": {
            "loss_job_count": len(loss_jobs),
            "loss_jobs": [
                {"id": r.job_id, "name": r.job_name, "loss_amount": abs(r.estimated_gross_profit), "contract": r.total_contract_price} 
                for r in loss_jobs
            ],
            "total_loss_exposure": total_loss_exposure
        },
        "margin_risk": {
            "fade_job_count": len(fade_jobs),
            "fade_jobs": fade_jobs[:5],  # Top 5 for brevity
            "uegp_at_risk": uegp_at_risk,
            "uegp_at_risk_pct": uegp_at_risk / total_uegp if total_uegp > 0 else 0
        },
        "concentration_risk": {
            "concentrated_jobs": [
                {"id": r.job_id, "name": r.job_name, "contract": r.total_contract_price, "pct_of_portfolio": r.total_contract_price / calc.total_contract_price if calc.total_contract_price else 0} 
                for r in concentration_jobs
            ]
        },
        "overbilling_note": {
            "severe_ob_count": len(severe_ob_jobs),
            "total_ob": sum(r.over_billings for r in severe_ob_jobs)
        }
    }

def compute_portfolio_risk_tier(risk_context: Dict[str, Any]) -> str:
    """Compute an overall risk tier for quick triage."""
    score = 0
    
    # Cash risk (most important for surety)
    if risk_context["cash_risk"]["total_ub_exposure"] > 500000:
        score += 4
    elif risk_context["cash_risk"]["total_ub_exposure"] > 100000:
        score += 2
    
    # Loss risk
    if risk_context["loss_risk"]["loss_job_count"] >= 3:
        score += 3
    elif risk_context["loss_risk"]["loss_job_count"] >= 1:
        score += 2
    
    # Margin erosion
    if risk_context["margin_risk"]["uegp_at_risk_pct"] > 0.3:
        score += 2
    elif risk_context["margin_risk"]["uegp_at_risk_pct"] > 0.15:
        score += 1
    
    # Concentration
    if len(risk_context["concentration_risk"]["concentrated_jobs"]) > 0:
        score += 1
    
    if score >= 6:
        return "HIGH"
    elif score >= 3:
        return "MODERATE"
    else:
        return "LOW"

def build_risk_rows(rows: List[CalculatedWipRow], calc: WipTotals) -> List[Dict[str, Any]]:
    """Build risk rows for the UI, sorted by surety priority."""
    risks = []
    
    for r in rows:
        job_risk_tags = []
        risk_details = []  # Detailed breakdown for dropdown
        risk_score = 0
        
        # Check for loss job (highest priority)
        if r.estimated_gross_profit < 0:
            job_risk_tags.append("Loss Job")
            risk_score += 50 + abs(r.estimated_gross_profit) / 1000
            risk_details.append({
                "tag": "Loss Job",
                "summary": f"Estimated GP: -${abs(r.estimated_gross_profit):,.0f}",
                "detail": "This job is projected to lose money. Every dollar of loss reduces the contractor's capacity to pay obligations."
            })
        
        # Check for severe underbilling - EITHER high % OR large absolute amount
        is_severe_ub = (
            (r.under_billings > r.total_contract_price * 0.10 and r.under_billings > 25000) or  # High % of contract
            (r.under_billings > 100000)  # Large absolute amount regardless of %
        )
        if is_severe_ub:
            job_risk_tags.append("Severe Underbilling")
            risk_score += 40 + r.under_billings / 10000
            ub_pct = (r.under_billings / r.total_contract_price * 100) if r.total_contract_price else 0
            risk_details.append({
                "tag": "Severe Underbilling",
                "summary": f"${r.under_billings:,.0f} underbilled ({ub_pct:.0f}% of contract)",
                "detail": "Work completed but cash not collected. If default occurs, this represents money that may never be recovered."
            })
        
        # Check for GP fade
        if r.estimated_total_costs > 0 and r.estimated_gross_profit > 0:
            poc = r.cost_to_date / r.estimated_total_costs
            expected_gp = r.estimated_gross_profit * poc
            if expected_gp > 1000 and r.gross_profit_to_date < expected_gp * 0.7:
                fade_pct = 1 - (r.gross_profit_to_date / expected_gp)
                job_risk_tags.append(f"GP Fade ({fade_pct:.0%})")
                risk_score += 20 + fade_pct * 30
                risk_details.append({
                    "tag": f"GP Fade ({fade_pct:.0%})",
                    "summary": f"Expected GP: ${expected_gp:,.0f} | Actual: ${r.gross_profit_to_date:,.0f}",
                    "detail": "Profit is lagging behind where it should be at this completion percentage. May indicate cost overruns or estimating problems."
                })
        
        # Check for thin margins
        if r.total_contract_price > 0 and 0 < r.estimated_gross_profit < r.total_contract_price * 0.05:
            margin_pct = (r.estimated_gross_profit / r.total_contract_price * 100)
            job_risk_tags.append("Thin Margin")
            risk_score += 10
            risk_details.append({
                "tag": "Thin Margin",
                "summary": f"GP margin only {margin_pct:.1f}% of contract value",
                "detail": "Very little buffer for cost increases or unforeseen issues. Small problems could push this job into loss territory."
            })
        
        # Moderate: overbilling (less critical but worth flagging)
        is_severe_ob = (
            (r.over_billings > r.total_contract_price * 0.15 and r.over_billings > 25000) or
            (r.over_billings > 100000)
        )
        if is_severe_ob:
            job_risk_tags.append("Heavy Overbilling")
            risk_score += 5
            ob_pct = (r.over_billings / r.total_contract_price * 100) if r.total_contract_price else 0
            risk_details.append({
                "tag": "Heavy Overbilling",
                "summary": f"${r.over_billings:,.0f} overbilled ({ob_pct:.0f}% of contract)",
                "detail": "Cash collected ahead of work performed. Good for recovery position, but can mask underlying problems."
            })
        
        # Check concentration (handled at portfolio level, but flag the job)
        if calc.total_contract_price > 0 and r.total_contract_price > calc.total_contract_price * 0.25:
            concentration_pct = (r.total_contract_price / calc.total_contract_price * 100)
            job_risk_tags.append("Concentration Risk")
            risk_score += 15
            risk_details.append({
                "tag": "Concentration Risk",
                "summary": f"Represents {concentration_pct:.0f}% of total portfolio value",
                "detail": "A single large job going bad could significantly impact the contractor's overall financial position."
            })
        
        if job_risk_tags:
            # Determine risk tier
            if risk_score >= 50:
                risk_tier = "HIGH"
            elif risk_score >= 20:
                risk_tier = "MEDIUM"
            else:
                risk_tier = "LOW"
            
            poc = r.cost_to_date / r.estimated_total_costs if r.estimated_total_costs else 0
            
            risks.append({
                "jobId": r.job_id,
                "jobName": r.job_name,
                "riskTags": ", ".join(job_risk_tags),
                "riskTier": risk_tier,
                "riskScore": risk_score,
                "riskDetails": risk_details,
                "percentComplete": f"{poc:.1%}",
                # Keep raw values for potential use
                "contractValue": r.total_contract_price,
                "underBillings": r.under_billings,
                "overBillings": r.over_billings
            })
    
    # Sort by risk score descending
    risks.sort(key=lambda x: x["riskScore"], reverse=True)
    
    return risks

# ==========================================
# 6. EXTRACTOR NODE (unchanged from original)
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
            except Exception as e:
                print(f"Warning: Could not parse totals row: {e}")
        
        return {"processed_data": rows, "totals_row": totals}
        
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {"processed_data": [], "totals_row": None}

# ==========================================
# 7. ANALYST NODE
# ==========================================

def analyst_node(state: WipState):
    print("--- RUNNING VALIDATIONS & ANALYSIS ---")
    
    try:
        rows = state.processed_data
        extracted_totals = state.totals_row
        
        if not rows:
            return {
                "final_json": {"error": "No data found"},
                "validation_errors": [],
                "correction_suggestions": [],
                "surety_risk_context": {},
                "risk_rows": [],
                "widget_data": {}
            }
        
        calc = WipTotals()
        validations = build_validations()
        all_validation_errors = []
        all_correction_suggestions = []
    
        # Recalculate UB/OB to ensure mathematical consistency
        for r in rows:
            variance = r.revenues_earned - r.billed_to_date
            if variance > 0:
                r.under_billings = variance
                r.over_billings = 0.0
            else:
                r.under_billings = 0.0
                r.over_billings = abs(variance)

        # AGGREGATION & VALIDATION
        for r in rows:
            # Aggregate totals
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
            
            # Run validations
            row_errors = run_validations(r, validations)
            all_validation_errors.extend(row_errors)
            
            # Generate correction suggestions
            if row_errors:
                suggestions = suggest_corrections(r, row_errors)
                all_correction_suggestions.extend(suggestions)

        # --- BUILD SURETY RISK CONTEXT ---
        surety_risk_context = build_surety_risk_context(rows, calc)
        surety_risk_context["risk_tier"] = compute_portfolio_risk_tier(surety_risk_context)
        
        # --- BUILD RISK ROWS ---
        risk_rows = build_risk_rows(rows, calc)

        # --- KPIs (preserved from original) ---
        t_uegp = calc.estimated_gross_profit - calc.gross_profit_to_date
        gp_percent = (calc.gross_profit_to_date / calc.revenues_earned * 100) if calc.revenues_earned else 0
        
        net_ub_ob = calc.under_billings - calc.over_billings
        net_ub_ob_label = f"Under ${net_ub_ob/1000:.0f}k" if net_ub_ob >= 0 else f"Over ${abs(net_ub_ob)/1000:.0f}k"

        # --- STRUCTURAL VALIDATION ---
        struct_pass = len(rows) > 0 and all(r.job_id for r in rows)
        struct_msg = "Structure Valid" if struct_pass else "Missing IDs/Data"
        
        # --- FORMULAIC VALIDATION SUMMARY ---
        # Group errors by job for display
        errors_by_job = {}
        for e in all_validation_errors:
            job_id = e["job_id"]
            if job_id not in errors_by_job:
                errors_by_job[job_id] = []
            errors_by_job[job_id].append(e)
        
        formula_errors_display = []
        for job_id, errors in errors_by_job.items():
            for e in errors:
                formula_errors_display.append({
                    "id": job_id,
                    "msg": e["message"]
                })
        
        formula_pass = len(all_validation_errors) == 0
        if formula_pass:
            formula_msg = "Column Math Validated"
        else:
            formula_msg = f"Column Math Issues ({len(errors_by_job)} rows)"

        # --- TOTALS VALIDATION ---
        totals_pass = False
        totals_msg = "No Totals Row"
        totals_details = []
        if extracted_totals:
            diff = abs(calc.revenues_earned - extracted_totals.revenues_earned)
            if diff < 1000.0:
                totals_pass = True
                totals_msg = "Sum matches Report Total"
            else:
                totals_msg = f"Sum Mismatch (${diff:,.0f})"
                totals_details.append({
                    "id": "TOTALS", 
                    "msg": f"Calc Earned ${calc.revenues_earned:,.0f} vs Report ${extracted_totals.revenues_earned:,.0f}"
                })

        # --- CORRECTION SUGGESTIONS SUMMARY (grouped by job) ---
        # Group errors and corrections by job for unified display
        jobs_with_issues = {}
        
        for e in all_validation_errors:
            job_id = e["job_id"]
            if job_id not in jobs_with_issues:
                jobs_with_issues[job_id] = {"errors": [], "corrections": []}
            jobs_with_issues[job_id]["errors"].append({
                "validation": e["validation"],
                "message": e["message"],
                "category": e["category"]
            })
        
        for s in all_correction_suggestions:
            job_id = s["job_id"]
            if job_id not in jobs_with_issues:
                jobs_with_issues[job_id] = {"errors": [], "corrections": []}
            jobs_with_issues[job_id]["corrections"].append({
                "field": s["field"],
                "current": f"${s['current_value']:,.0f}",
                "suggested": f"${s['suggested_value']:,.0f}",
                "confidence": s["confidence"],
                "reasoning": s["reasoning"]
            })
        
        # Convert to list format for frontend
        corrections_display = []
        for job_id, data in jobs_with_issues.items():
            corrections_display.append({
                "jobId": job_id,
                "errors": data["errors"],
                "corrections": data["corrections"]
            })

        # --- BUILD WIDGET DATA ---
        widget_data = {
            "validations": {
                "structural": {"passed": struct_pass, "message": struct_msg, "details": []},
                "formulaic": {"passed": formula_pass, "message": formula_msg, "details": formula_errors_display, "jobIssues": corrections_display},
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
            "riskTier": surety_risk_context["risk_tier"],
            "riskRowsAll": risk_rows
        }

        return {
            "calculated_totals": calc,
            "validation_errors": all_validation_errors,
            "correction_suggestions": all_correction_suggestions,
            "surety_risk_context": surety_risk_context,
            "risk_rows": risk_rows,
            "widget_data": widget_data
        }
    
    except Exception as e:
        print(f"ANALYST NODE ERROR: {e}")
        print(traceback.format_exc())
        # Return minimal valid response so pipeline continues
        return {
            "calculated_totals": WipTotals(),
            "validation_errors": [],
            "correction_suggestions": [],
            "surety_risk_context": {"risk_tier": "UNKNOWN", "error": str(e)},
            "risk_rows": [],
            "widget_data": {
                "validations": {
                    "structural": {"passed": False, "message": f"Error: {str(e)}", "details": []},
                    "formulaic": {"passed": False, "message": "Error during analysis", "details": [], "jobIssues": []},
                    "totals": {"passed": False, "message": "Error during analysis", "details": []}
                },
                "metrics": {
                    "row1_1": {"label": "Contract Value", "value": "$0.00M"},
                    "row1_2": {"label": "UEGP", "value": "$0.00M"},
                    "row1_3": {"label": "CTC", "value": "$0.00M"},
                    "row2_1": {"label": "Earned Rev", "value": "$0.00M"},
                    "row2_2": {"label": "GP %", "value": "0.0%"},
                    "row2_3": {"label": "Net UB / OB", "value": "Error"}
                },
                "riskTier": "UNKNOWN",
                "riskRowsAll": [],
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }

# ==========================================
# 8. NARRATIVE NODE
# ==========================================

SURETY_NARRATIVE_PROMPT = """
You are a surety underwriting analyst writing a brief summary for a senior underwriter.

CONTEXT: We bonded this contractor. If they default, we need to recover what we're owed. Our concerns, in priority order:

1. CASH POSITION — Severe underbilling means work done but cash not collected. This is our biggest concern because it directly impacts recovery.
2. LOSS JOBS — Negative margin jobs burn cash and reduce their ability to pay us.
3. MARGIN EROSION — Jobs where profit is fading signal estimating problems or emerging losses.
4. CONCENTRATION — One big job going bad can sink everything.

Overbilling is less urgent—it means they've collected ahead of work, which is actually better for our recovery position (though it can mask problems). Mention it only if extreme.

INSTRUCTIONS:
- Write 3-4 sentences maximum
- Lead with the most important concern for recovery risk
- Use plain language, no jargon
- Be direct about problems; don't soften bad news
- If the portfolio looks healthy, say so briefly and note any minor watch items
- Include specific dollar amounts and job counts where relevant

RISK ASSESSMENT:
{risk_context}

SUMMARY:"""

def narrative_node(state: WipState):
    print("--- GENERATING NARRATIVE ---")
    
    try:
        risk_context = state.surety_risk_context
        
        if not risk_context:
            print("No risk context available for narrative")
            widget_data = state.widget_data.copy() if state.widget_data else {}
            widget_data["summary"] = {"text": "Unable to generate narrative: no risk context available."}
            return {"narrative": "Unable to generate narrative: no risk context available.", "widget_data": widget_data}
        
        # Check for error in risk context (from analyst node failure)
        if "error" in risk_context:
            print(f"Risk context contains error: {risk_context.get('error')}")
            widget_data = state.widget_data.copy() if state.widget_data else {}
            widget_data["summary"] = {"text": f"Analysis error: {risk_context.get('error')}"}
            return {"narrative": f"Analysis error: {risk_context.get('error')}", "widget_data": widget_data}
        
        prompt = SURETY_NARRATIVE_PROMPT.format(risk_context=json.dumps(risk_context, indent=2))
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=250
            )
        )
        
        narrative = response.text.strip()
        
        # Update widget_data with narrative
        widget_data = state.widget_data.copy() if state.widget_data else {}
        widget_data["summary"] = {"text": narrative}
        
        return {"narrative": narrative, "widget_data": widget_data}
        
    except Exception as e:
        print(f"NARRATIVE NODE ERROR: {e}")
        print(traceback.format_exc())
        
        # Fallback to basic summary
        try:
            risk_context = state.surety_risk_context or {}
            portfolio = risk_context.get("portfolio", {})
            total_value = portfolio.get('total_contract_value', 0)
            fallback = (
                f"Portfolio contains {portfolio.get('total_jobs', 0)} jobs with "
                f"${total_value/1000000:.1f}M total contract value. "
                f"Risk tier: {risk_context.get('risk_tier', 'Unknown')}."
            )
        except:
            fallback = "Error generating narrative summary."
        
        widget_data = state.widget_data.copy() if state.widget_data else {}
        widget_data["summary"] = {"text": fallback}
        widget_data["narrative_error"] = str(e)
        
        return {"narrative": fallback, "widget_data": widget_data}

# ==========================================
# 9. OUTPUT NODE
# ==========================================

def output_node(state: WipState):
    print("--- BUILDING FINAL OUTPUT ---")
    
    try:
        payload = {
            "clean_table": [r.model_dump() for r in state.processed_data] if state.processed_data else [],
            "calculated_totals": state.calculated_totals.model_dump() if state.calculated_totals else {},
            "validation_errors": state.validation_errors or [],
            "correction_suggestions": state.correction_suggestions or [],
            "surety_risk_context": state.surety_risk_context or {},
            "widget_data": state.widget_data or {}
        }
        
        return {"final_json": payload}
    
    except Exception as e:
        print(f"OUTPUT NODE ERROR: {e}")
        print(traceback.format_exc())
        return {
            "final_json": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "clean_table": [],
                "calculated_totals": {},
                "validation_errors": [],
                "correction_suggestions": [],
                "surety_risk_context": {},
                "widget_data": {"error": str(e)}
            }
        }

# ==========================================
# 10. WORKFLOW
# ==========================================

workflow = StateGraph(WipState)
workflow.add_node("extract", extractor_node)
workflow.add_node("analyze", analyst_node)
workflow.add_node("narrative", narrative_node)
workflow.add_node("output", output_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", "narrative")
workflow.add_edge("narrative", "output")
workflow.add_edge("output", END)

app = workflow.compile()
