# =====================
# wip_agent.py (UPDATED)
# =====================
from __future__ import annotations

import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field, computed_field
from langgraph.graph import StateGraph, END

# Model client abstraction
from model_client import (
    get_client,
    MetricsTracker,
    DEFAULT_MODEL,
    parse_json_safely,
)

logger = logging.getLogger(__name__)

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

    # Extracted from document (as-is)
    under_billings: float = Field(default=0.0)
    over_billings: float = Field(default=0.0)

    # Calculated from earned vs billed (populated in analyst_node)
    under_billings_calc: float = Field(default=0.0, description="Calculated UB from revenues_earned - billed_to_date")
    over_billings_calc: float = Field(default=0.0, description="Calculated OB from billed_to_date - revenues_earned")


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

    @computed_field
    @property
    def billing_variance(self) -> float:
        return self.revenues_earned - self.billed_to_date

    @computed_field
    @property
    def ub_ob_discrepancy_abs(self) -> float:
        """How far the extracted UB/OB differs from the computed UB/OB."""
        return abs(self.under_billings - self.under_billings_calc) + abs(self.over_billings - self.over_billings_calc)


class WipState(BaseModel):
    file_path: str
    model_name: str = DEFAULT_MODEL

    # Metrics tracker stored as dict for Pydantic serialization
    metrics_data: Dict[str, Any] = Field(default_factory=dict)

    processed_data: List[CalculatedWipRow] = Field(default_factory=list)
    totals_row: Optional[WipTotals] = None
    calculated_totals: Optional[WipTotals] = None

    validation_errors: List[Dict[str, Any]] = Field(default_factory=list)
    correction_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    correction_log: List[Dict[str, Any]] = Field(default_factory=list)
    extraction_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    surety_risk_context: Dict[str, Any] = Field(default_factory=dict)
    risk_rows: List[Dict[str, Any]] = Field(default_factory=list)
    widget_data: Dict[str, Any] = Field(default_factory=dict)

    narrative: str = ""
    final_json: Dict[str, Any] = Field(default_factory=dict)

# ==========================================
# 3. VALIDATION REGISTRY
# ==========================================

@dataclass
class Validation:
    name: str
    requires: List[str]
    check: Callable[[CalculatedWipRow], Optional[str]]
    tolerance_type: str  # "absolute" or "percentage" (kept for future framework refactor)
    tolerance_value: float
    category: str  # "core_math", "billing", "profitability", "completion"
    fields_involved: List[str]  # Fields that could be the source of error


def _check_billing_position(r: CalculatedWipRow) -> Optional[str]:
    """Validates that extracted UB/OB values match the earned revenue vs billed calculation."""
    variance = r.revenues_earned - r.billed_to_date

    if variance > 0:
        expected_ub, expected_ob = variance, 0.0
    else:
        expected_ub, expected_ob = 0.0, abs(variance)

    ub_diff = abs(r.under_billings - expected_ub)
    ob_diff = abs(r.over_billings - expected_ob)

    if ub_diff > 100 or ob_diff > 100:
        return f"UB/OB doesn't match variance (expected UB ${expected_ub:,.0f}, OB ${expected_ob:,.0f})"
    return None


def build_validations() -> List[Validation]:
    return [
        Validation(
            name="contract_cost_gp",
            requires=["total_contract_price", "estimated_total_costs", "estimated_gross_profit"],
            check=lambda r: (
                f"Contract - Est Cost != Est GP (diff ${abs((r.total_contract_price - r.estimated_total_costs) - r.estimated_gross_profit):,.0f})"
                if abs((r.total_contract_price - r.estimated_total_costs) - r.estimated_gross_profit) > 100
                else None
            ),
            tolerance_type="absolute",
            tolerance_value=100,
            category="core_math",
            fields_involved=["total_contract_price", "estimated_total_costs", "estimated_gross_profit"],
        ),
        Validation(
            name="cost_to_complete_check",
            requires=["estimated_total_costs", "cost_to_date", "cost_to_complete"],
            check=lambda r: (
                f"Est Cost - Cost to Date != CTC (diff ${abs((r.estimated_total_costs - r.cost_to_date) - r.cost_to_complete):,.0f})"
                if abs((r.estimated_total_costs - r.cost_to_date) - r.cost_to_complete) > 100
                else None
            ),
            tolerance_type="absolute",
            tolerance_value=100,
            category="core_math",
            fields_involved=["estimated_total_costs", "cost_to_date", "cost_to_complete"],
        ),
        Validation(
            name="earned_revenue_from_gp",
            requires=["revenues_earned", "cost_to_date", "gross_profit_to_date"],
            check=lambda r: (
                f"Cost + GP to Date != Earned Rev (diff ${abs(r.revenues_earned - (r.cost_to_date + r.gross_profit_to_date)):,.0f})"
                if abs(r.revenues_earned - (r.cost_to_date + r.gross_profit_to_date)) > 100
                else None
            ),
            tolerance_type="absolute",
            tolerance_value=100,
            category="core_math",
            fields_involved=["revenues_earned", "cost_to_date", "gross_profit_to_date"],
        ),
        Validation(
            name="remaining_gp_check",
            requires=["estimated_gross_profit", "gross_profit_to_date"],
            check=lambda r: (
                "UEGP calculation mismatch"
                if r.estimated_gross_profit > 0 and r.gross_profit_to_date > r.estimated_gross_profit * 1.05
                else None
            ),
            tolerance_type="percentage",
            tolerance_value=0.05,
            category="core_math",
            fields_involved=["estimated_gross_profit", "gross_profit_to_date"],
        ),
        Validation(
            name="earned_revenue_from_poc",
            requires=["revenues_earned", "total_contract_price", "cost_to_date", "estimated_total_costs"],
            check=lambda r: (
                f"Earned Rev != Contract x POC (diff ${abs(r.revenues_earned - (r.total_contract_price * (r.cost_to_date / r.estimated_total_costs if r.estimated_total_costs else 0))):,.0f})"
                if r.estimated_total_costs
                and abs(r.revenues_earned - (r.total_contract_price * (r.cost_to_date / r.estimated_total_costs)))
                > max(5000, r.total_contract_price * 0.02)
                else None
            ),
            tolerance_type="percentage",
            tolerance_value=0.02,
            category="completion",
            fields_involved=["revenues_earned", "total_contract_price", "cost_to_date", "estimated_total_costs"],
        ),
        Validation(
            name="underbilling_overbilling",
            requires=["revenues_earned", "billed_to_date", "under_billings", "over_billings"],
            check=_check_billing_position,
            tolerance_type="absolute",
            tolerance_value=100,
            category="billing",
            fields_involved=["revenues_earned", "billed_to_date", "under_billings", "over_billings"],
        ),
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
            fields_involved=["estimated_gross_profit", "total_contract_price"],
        ),
    ]


def run_validations(row: CalculatedWipRow, validations: List[Validation]) -> List[Dict[str, Any]]:
    """Run all applicable validations for a row."""
    errors: List[Dict[str, Any]] = []
    row_dict = row.model_dump()

    for v in validations:
        required_present = all(row_dict.get(field) is not None for field in v.requires)
        has_nonzero = any(row_dict.get(field, 0) != 0 for field in v.requires)

        if not required_present or not has_nonzero:
            continue

        error_msg = v.check(row)
        if error_msg:
            errors.append(
                {
                    "job_id": row.job_id,
                    "validation": v.name,
                    "category": v.category,
                    "message": error_msg,
                    "fields_involved": v.fields_involved,
                }
            )

    return errors

# ==========================================
# 4. CORRECTION SUGGESTIONS (unchanged)
# ==========================================

def digit_change_score(current: float, target: float) -> int:
    """
    Score how many digit-level changes separate two numbers.
    Lower = more likely a single OCR/extraction misread.

    Handles:
    - Same-length digit misreads (e.g., 1234 → 1834)
    - Dropped/added single digit (e.g., 150000 → 15000, 11000 → 1100)
    - Sign misreads (parenthetical negatives parsed as positive)
    """
    if current == 0 and target == 0:
        return 0
    if current == 0 or target == 0:
        return 100

    curr_str = str(int(abs(current)))
    targ_str = str(int(abs(target)))

    sign_penalty = 50 if (current < 0) != (target < 0) else 0

    len_diff = abs(len(curr_str) - len(targ_str))

    # Dropped/added single digit (including trailing zeros)
    # e.g., 150000 → 15000, 11000 → 1100, 234567 → 23456
    if len_diff == 1:
        longer = curr_str if len(curr_str) > len(targ_str) else targ_str
        shorter = targ_str if len(curr_str) > len(targ_str) else curr_str
        for i in range(len(longer)):
            if longer[:i] + longer[i + 1:] == shorter:
                return 1 + sign_penalty  # Single digit drop/insert
        # No single removal works — fall through to padded comparison

    if len_diff > 1:
        return 30 + len_diff * 10 + sign_penalty

    # Same length (or len_diff==1 where no single removal matched): pad and count
    max_len = max(len(curr_str), len(targ_str))
    curr_padded = curr_str.zfill(max_len)
    targ_padded = targ_str.zfill(max_len)

    digit_diffs = sum(1 for a, b in zip(curr_padded, targ_padded) if a != b)

    return digit_diffs + sign_penalty

def find_best_fix(candidates: List[Dict]) -> Optional[Dict]:
    if not candidates:
        return None

    scored = []
    for c in candidates:
        score = digit_change_score(c["current"], c["suggested"])
        scored.append((score, c))

    scored.sort(key=lambda x: x[0])
    best_score, best = scored[0]

    best["digit_changes"] = best_score
    best["confidence"] = "high" if best_score <= 2 else "medium" if best_score <= 4 else "low"
    return best


def _get_candidates_for_validation(row: CalculatedWipRow, validation_name: str) -> List[Dict]:
    """Generate algebraic fix candidates for a given validation failure."""
    candidates = []

    if validation_name == "contract_cost_gp":
        expected_gp = row.total_contract_price - row.estimated_total_costs
        expected_cost = row.total_contract_price - row.estimated_gross_profit
        expected_contract = row.estimated_total_costs + row.estimated_gross_profit
        candidates = [
            {"field": "estimated_gross_profit", "current": row.estimated_gross_profit, "suggested": expected_gp, "formula": "Contract - Est Cost = Est GP"},
            {"field": "estimated_total_costs", "current": row.estimated_total_costs, "suggested": expected_cost, "formula": "Contract - Est Cost = Est GP"},
            {"field": "total_contract_price", "current": row.total_contract_price, "suggested": expected_contract, "formula": "Contract - Est Cost = Est GP"},
        ]

    elif validation_name == "cost_to_complete_check":
        expected_ctc = row.estimated_total_costs - row.cost_to_date
        expected_cost = row.cost_to_complete + row.cost_to_date
        expected_ctd = row.estimated_total_costs - row.cost_to_complete
        candidates = [
            {"field": "cost_to_complete", "current": row.cost_to_complete, "suggested": expected_ctc, "formula": "Est Cost - Cost to Date = CTC"},
            {"field": "estimated_total_costs", "current": row.estimated_total_costs, "suggested": expected_cost, "formula": "Est Cost - Cost to Date = CTC"},
            {"field": "cost_to_date", "current": row.cost_to_date, "suggested": expected_ctd, "formula": "Est Cost - Cost to Date = CTC"},
        ]

    elif validation_name == "earned_revenue_from_gp":
        expected_rev = row.cost_to_date + row.gross_profit_to_date
        expected_gp_td = row.revenues_earned - row.cost_to_date
        expected_ctd = row.revenues_earned - row.gross_profit_to_date
        candidates = [
            {"field": "revenues_earned", "current": row.revenues_earned, "suggested": expected_rev, "formula": "Cost to Date + GP to Date = Earned Rev"},
            {"field": "gross_profit_to_date", "current": row.gross_profit_to_date, "suggested": expected_gp_td, "formula": "Cost to Date + GP to Date = Earned Rev"},
            {"field": "cost_to_date", "current": row.cost_to_date, "suggested": expected_ctd, "formula": "Cost to Date + GP to Date = Earned Rev"},
        ]

    elif validation_name == "earned_revenue_from_poc":
        if row.estimated_total_costs > 0:
            poc = row.cost_to_date / row.estimated_total_costs
            expected_rev = row.total_contract_price * poc
            candidates = [{"field": "revenues_earned", "current": row.revenues_earned, "suggested": expected_rev, "formula": "Earned Rev = Contract x POC"}]

    return candidates


def suggest_corrections(row: CalculatedWipRow, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate correction suggestions with cross-validation ranking.

    When multiple validations implicate the same field, that field's score
    is reduced (= more likely the real culprit).
    """
    # Collect ALL candidates across ALL errors for this row
    all_candidates: List[tuple] = []  # (validation_name, candidate_dict)
    for error in errors:
        candidates = _get_candidates_for_validation(row, error["validation"])
        for c in candidates:
            all_candidates.append((error["validation"], c))

    # Count how many distinct validations implicate each field
    field_validation_count: Dict[str, int] = {}
    seen: Dict[str, set] = {}
    for vname, c in all_candidates:
        seen.setdefault(c["field"], set()).add(vname)
    field_validation_count = {f: len(v) for f, v in seen.items()}

    # Score and pick best fix per error
    suggestions: List[Dict[str, Any]] = []

    for error in errors:
        candidates = _get_candidates_for_validation(row, error["validation"])
        if not candidates:
            continue

        scored = []
        for c in candidates:
            base_score = digit_change_score(c["current"], c["suggested"])
            cross_count = field_validation_count.get(c["field"], 1)
            # Cross-validation boost: each additional validation agreeing reduces score by 1
            effective_score = max(0, base_score - (cross_count - 1))
            scored.append((effective_score, base_score, c, cross_count))

        scored.sort(key=lambda x: x[0])
        effective, raw, best, cross_count = scored[0]

        confidence = "high" if effective <= 2 else "medium" if effective <= 4 else "low"

        suggestions.append(
            {
                "job_id": row.job_id,
                "field": best["field"],
                "current_value": best["current"],
                "suggested_value": best["suggested"],
                "confidence": confidence,
                "digit_changes": raw,
                "effective_score": effective,
                "cross_validation_count": cross_count,
                "reasoning": (
                    f"Changing {best['field']} from ${best['current']:,.0f} to ${best['suggested']:,.0f} "
                    f"({raw} digit change"
                    + (f", {cross_count} validations agree" if cross_count > 1 else "")
                    + f") would fix: {best['formula']}"
                ),
            }
        )

    return suggestions


# ==========================================
# 4b. COLUMN SWAP DETECTION (Tier 1)
# ==========================================

def detect_column_swaps(rows: List[CalculatedWipRow]) -> List[Dict[str, Any]]:
    """
    Detect likely Cost to Date <-> Cost to Complete column swaps.

    Signal: if revenue-based percent complete (revenues_earned / total_contract_price)
    diverges significantly from cost-based percent complete (cost_to_date / estimated_total_costs),
    AND swapping CTD/CTC would bring them into alignment, the columns are probably swapped.
    """
    swaps: List[Dict[str, Any]] = []
    for r in rows:
        if r.estimated_total_costs <= 0 or r.total_contract_price <= 0:
            continue
        if r.cost_to_date <= 0 and r.cost_to_complete <= 0:
            continue

        revenue_poc = r.revenues_earned / r.total_contract_price
        cost_poc = r.cost_to_date / r.estimated_total_costs

        # Revenue suggests far along but cost suggests early, and CTC > CTD
        if (revenue_poc > 0.5 and cost_poc < 0.5
                and r.cost_to_complete > r.cost_to_date
                and r.cost_to_complete > 0):

            swapped_poc = r.cost_to_complete / r.estimated_total_costs
            # Swap only if it brings cost POC closer to revenue POC
            if abs(swapped_poc - revenue_poc) < abs(cost_poc - revenue_poc):
                swaps.append({
                    "job_id": r.job_id,
                    "type": "column_swap",
                    "fields": ["cost_to_date", "cost_to_complete"],
                    "original_ctd": r.cost_to_date,
                    "original_ctc": r.cost_to_complete,
                    "confidence": "high",
                    "reasoning": (
                        f"Revenue POC ({revenue_poc:.0%}) vs Cost POC ({cost_poc:.0%}) mismatch — "
                        f"swapping CTD/CTC aligns to {swapped_poc:.0%}"
                    ),
                })

    return swaps


def apply_corrections(
    rows: List[CalculatedWipRow],
    column_swaps: List[Dict[str, Any]],
    digit_corrections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Apply high-confidence corrections in-place and return a log of what changed.

    Tier 1: Column swaps (always applied).
    Tier 2: Digit corrections with 'high' confidence.
    """
    correction_log: List[Dict[str, Any]] = []

    swap_jobs = {s["job_id"] for s in column_swaps}

    # Index digit corrections by job_id (high confidence only)
    digit_fixes: Dict[str, List[Dict]] = {}
    for d in digit_corrections:
        if d.get("confidence") == "high":
            digit_fixes.setdefault(d["job_id"], []).append(d)

    for r in rows:
        # Apply column swaps
        if r.job_id in swap_jobs:
            old_ctd, old_ctc = r.cost_to_date, r.cost_to_complete
            r.cost_to_date = old_ctc
            r.cost_to_complete = old_ctd
            correction_log.append({
                "job_id": r.job_id,
                "type": "column_swap",
                "action": f"Swapped CTD (${old_ctd:,.0f}) ↔ CTC (${old_ctc:,.0f})",
                "confidence": "high",
            })

        # Apply digit corrections
        if r.job_id in digit_fixes:
            for fix in digit_fixes[r.job_id]:
                field = fix["field"]
                old_val = getattr(r, field, None)
                if old_val is not None:
                    setattr(r, field, fix["suggested_value"])
                    correction_log.append({
                        "job_id": r.job_id,
                        "type": "digit_fix",
                        "field": field,
                        "action": f"Changed {field} from ${old_val:,.0f} to ${fix['suggested_value']:,.0f}",
                        "confidence": fix["confidence"],
                        "reasoning": fix["reasoning"],
                    })

    return correction_log

# ==========================================
# 5. SURETY RISK ANALYSIS (updated to use *_calc)
# ==========================================

def build_surety_risk_context(rows: List[CalculatedWipRow], calc: WipTotals) -> Dict[str, Any]:
    loss_jobs = [r for r in rows if r.estimated_gross_profit < 0]
    total_loss_exposure = sum(abs(r.estimated_gross_profit) for r in loss_jobs)

    severe_ub_jobs = [
        r for r in rows
        if ((r.under_billings_calc > r.total_contract_price * 0.10 and r.under_billings_calc > 25000) or (r.under_billings_calc > 100000))
    ]
    total_ub_exposure = sum(r.under_billings_calc for r in severe_ub_jobs)

    fade_jobs = []
    for r in rows:
        if r.estimated_total_costs > 0 and r.estimated_gross_profit > 0:
            poc = r.cost_to_date / r.estimated_total_costs
            expected_gp_to_date = r.estimated_gross_profit * poc
            if expected_gp_to_date > 1000 and r.gross_profit_to_date < expected_gp_to_date * 0.7:
                fade_jobs.append(
                    {
                        "job_id": r.job_id,
                        "job_name": r.job_name,
                        "expected_gp": expected_gp_to_date,
                        "actual_gp": r.gross_profit_to_date,
                        "fade_pct": 1 - (r.gross_profit_to_date / expected_gp_to_date) if expected_gp_to_date else 0,
                        "contract": r.total_contract_price,
                    }
                )

    concentration_jobs = [
        r for r in rows
        if calc.total_contract_price > 0 and r.total_contract_price > calc.total_contract_price * 0.25
    ]

    total_uegp = calc.estimated_gross_profit - calc.gross_profit_to_date
    thin_margin_jobs = [r for r in rows if r.total_contract_price > 0 and r.estimated_gross_profit < r.total_contract_price * 0.05]
    uegp_at_risk = sum(
        r.estimated_gross_profit - r.gross_profit_to_date
        for r in thin_margin_jobs
        if r.estimated_gross_profit > r.gross_profit_to_date
    )

    severe_ob_jobs = [
        r for r in rows
        if ((r.over_billings_calc > r.total_contract_price * 0.15 and r.over_billings_calc > 25000) or (r.over_billings_calc > 100000))
    ]

    # UB/OB extraction discrepancy signal (useful for diagnosing extraction vs accounting weirdness)
    ub_ob_mismatch_jobs = [r for r in rows if r.ub_ob_discrepancy_abs > 100]

    return {
        "portfolio": {
            "total_jobs": len(rows),
            "total_contract_value": calc.total_contract_price,
            "aggregate_poc": calc.cost_to_date / calc.estimated_total_costs if calc.estimated_total_costs else 0,
            "net_billing_position": calc.under_billings - calc.over_billings,
            "total_uegp": total_uegp,
            "total_gp_margin": calc.estimated_gross_profit / calc.total_contract_price if calc.total_contract_price else 0,
        },
        "cash_risk": {
            "severe_ub_count": len(severe_ub_jobs),
            "severe_ub_jobs": [
                {"id": r.job_id, "name": r.job_name, "ub_amount": r.under_billings_calc, "contract": r.total_contract_price}
                for r in severe_ub_jobs
            ],
            "total_ub_exposure": total_ub_exposure,
        },
        "loss_risk": {
            "loss_job_count": len(loss_jobs),
            "loss_jobs": [
                {"id": r.job_id, "name": r.job_name, "loss_amount": abs(r.estimated_gross_profit), "contract": r.total_contract_price}
                for r in loss_jobs
            ],
            "total_loss_exposure": total_loss_exposure,
        },
        "margin_risk": {
            "fade_job_count": len(fade_jobs),
            "fade_jobs": fade_jobs[:5],
            "uegp_at_risk": uegp_at_risk,
            "uegp_at_risk_pct": uegp_at_risk / total_uegp if total_uegp > 0 else 0,
        },
        "concentration_risk": {
            "concentrated_jobs": [
                {
                    "id": r.job_id,
                    "name": r.job_name,
                    "contract": r.total_contract_price,
                    "pct_of_portfolio": r.total_contract_price / calc.total_contract_price if calc.total_contract_price else 0,
                }
                for r in concentration_jobs
            ]
        },
        "overbilling_note": {
            "severe_ob_count": len(severe_ob_jobs),
            "total_ob": sum(r.over_billings_calc for r in severe_ob_jobs),
        },
        "extraction_signals": {
            "ub_ob_mismatch_count": len(ub_ob_mismatch_jobs),
            "ub_ob_mismatch_jobs": [
                {"id": r.job_id, "name": r.job_name, "discrepancy_abs": r.ub_ob_discrepancy_abs}
                for r in ub_ob_mismatch_jobs[:10]
            ],
        },
    }

def compute_portfolio_risk_tier(risk_context: Dict[str, Any]) -> str:
    score = 0

    if risk_context["cash_risk"]["total_ub_exposure"] > 500000:
        score += 4
    elif risk_context["cash_risk"]["total_ub_exposure"] > 100000:
        score += 2

    if risk_context["loss_risk"]["loss_job_count"] >= 3:
        score += 3
    elif risk_context["loss_risk"]["loss_job_count"] >= 1:
        score += 2

    if risk_context["margin_risk"]["uegp_at_risk_pct"] > 0.3:
        score += 2
    elif risk_context["margin_risk"]["uegp_at_risk_pct"] > 0.15:
        score += 1

    if len(risk_context["concentration_risk"]["concentrated_jobs"]) > 0:
        score += 1

    if score >= 6:
        return "HIGH"
    elif score >= 3:
        return "MODERATE"
    else:
        return "LOW"

def build_risk_rows(rows: List[CalculatedWipRow], calc: WipTotals) -> List[Dict[str, Any]]:
    risks: List[Dict[str, Any]] = []

    for r in rows:
        job_risk_tags = []
        risk_details = []
        risk_score = 0

        # Use calculated UB/OB for risk tagging (more reliable for surety lens)
        ub = r.under_billings_calc
        ob = r.over_billings_calc

        if r.estimated_gross_profit < 0:
            job_risk_tags.append("Loss Job")
            risk_score += 50 + abs(r.estimated_gross_profit) / 1000
            risk_details.append(
                {
                    "tag": "Loss Job",
                    "summary": f"Estimated GP: -${abs(r.estimated_gross_profit):,.0f}",
                    "detail": "This job is projected to lose money. Every dollar of loss reduces the contractor's capacity to pay obligations.",
                }
            )

        is_severe_ub = ((ub > r.total_contract_price * 0.10 and ub > 25000) or (ub > 100000))
        if is_severe_ub:
            job_risk_tags.append("Severe Underbilling")
            risk_score += 40 + ub / 10000
            ub_pct = (ub / r.total_contract_price * 100) if r.total_contract_price else 0
            risk_details.append(
                {
                    "tag": "Severe Underbilling",
                    "summary": f"${ub:,.0f} underbilled ({ub_pct:.0f}% of contract)",
                    "detail": "Work completed but cash not collected. If default occurs, this represents money that may never be recovered.",
                }
            )

        if r.estimated_total_costs > 0 and r.estimated_gross_profit > 0:
            poc = r.cost_to_date / r.estimated_total_costs
            expected_gp = r.estimated_gross_profit * poc
            if expected_gp > 1000 and r.gross_profit_to_date < expected_gp * 0.7:
                fade_pct = 1 - (r.gross_profit_to_date / expected_gp)
                job_risk_tags.append(f"GP Fade ({fade_pct:.0%})")
                risk_score += 20 + fade_pct * 30
                risk_details.append(
                    {
                        "tag": f"GP Fade ({fade_pct:.0%})",
                        "summary": f"Expected GP: ${expected_gp:,.0f} | Actual: ${r.gross_profit_to_date:,.0f}",
                        "detail": "Profit is lagging behind where it should be at this completion percentage. May indicate cost overruns or estimating problems.",
                    }
                )

        if r.total_contract_price > 0 and 0 < r.estimated_gross_profit < r.total_contract_price * 0.05:
            margin_pct = (r.estimated_gross_profit / r.total_contract_price * 100)
            job_risk_tags.append("Thin Margin")
            risk_score += 10
            risk_details.append(
                {
                    "tag": "Thin Margin",
                    "summary": f"GP margin only {margin_pct:.1f}% of contract value",
                    "detail": "Very little buffer for cost increases or unforeseen issues. Small problems could push this job into loss territory.",
                }
            )

        is_severe_ob = ((ob > r.total_contract_price * 0.15 and ob > 25000) or (ob > 100000))
        if is_severe_ob:
            job_risk_tags.append("Heavy Overbilling")
            risk_score += 5
            ob_pct = (ob / r.total_contract_price * 100) if r.total_contract_price else 0
            risk_details.append(
                {
                    "tag": "Heavy Overbilling",
                    "summary": f"${ob:,.0f} overbilled ({ob_pct:.0f}% of contract)",
                    "detail": "Cash collected ahead of work performed. Good for recovery position, but can mask underlying problems.",
                }
            )

        if calc.total_contract_price > 0 and r.total_contract_price > calc.total_contract_price * 0.25:
            concentration_pct = (r.total_contract_price / calc.total_contract_price * 100)
            job_risk_tags.append("Concentration Risk")
            risk_score += 15
            risk_details.append(
                {
                    "tag": "Concentration Risk",
                    "summary": f"Represents {concentration_pct:.0f}% of total portfolio value",
                    "detail": "A single large job going bad could significantly impact the contractor's overall financial position.",
                }
            )

        if job_risk_tags:
            if risk_score >= 50:
                risk_tier = "HIGH"
            elif risk_score >= 20:
                risk_tier = "MEDIUM"
            else:
                risk_tier = "LOW"

            poc = r.cost_to_date / r.estimated_total_costs if r.estimated_total_costs else 0

            risks.append(
                {
                    "jobId": r.job_id,
                    "jobName": r.job_name,
                    "riskTags": ", ".join(job_risk_tags),
                    "riskTier": risk_tier,
                    "riskScore": risk_score,
                    "riskDetails": risk_details,
                    "percentComplete": f"{poc:.1%}",
                    "contractValue": r.total_contract_price,
                    # Calculated (preferred for risk)
                    "underBillings": ub,
                    "overBillings": ob,
                    # Extracted (debug signal)
                    "underBillingsExtracted": r.under_billings,
                    "overBillingsExtracted": r.over_billings,
                    "ubObDiscrepancyAbs": r.ub_ob_discrepancy_abs,
                }
            )

    risks.sort(key=lambda x: x["riskScore"], reverse=True)
    return risks

# ==========================================
# 6. EXTRACTOR NODE
# ==========================================

def extractor_node(state: WipState):
    print(f"\n--- EXTRACTING DATA FROM: {state.file_path} ---")
    print(f"--- USING MODEL: {state.model_name} ---")

    tracker = MetricsTracker(model_name=state.model_name)
    client = get_client()
    diagnostics: Dict[str, Any] = {"model": state.model_name, "stages": []}

    def _diag(stage: str, status: str, detail: str = ""):
        diagnostics["stages"].append({"stage": stage, "status": status, "detail": detail})

    try:
        with open(state.file_path, "rb") as f:
            file_bytes = f.read()
    except FileNotFoundError:
        _diag("file_read", "FAIL", f"File not found: {state.file_path}")
        return {"processed_data": [], "totals_row": None, "metrics_data": tracker.get_metrics(), "extraction_diagnostics": diagnostics}
    except Exception as e:
        _diag("file_read", "FAIL", str(e))
        return {"processed_data": [], "totals_row": None, "metrics_data": tracker.get_metrics(), "extraction_diagnostics": diagnostics}

    _diag("file_read", "OK", f"{len(file_bytes)} bytes")

    prompt = """
    Extract the WIP Schedule table.
    1. Extract every job row with all financial columns, Job Name, and Job ID.
    2. Extract the "TOTALS" row from the bottom of the report.

    CRITICAL COLUMN IDENTIFICATION:

    1. COST TO DATE (cost_to_date): Money ALREADY SPENT on the job.
       - Column headers: "Cost to Date", "Costs to Date", "Costs Incurred", "Actual Costs"
       - This is cumulative costs incurred so far
       - Usually a LARGER number than Cost to Complete for jobs in progress

    2. COST TO COMPLETE (cost_to_complete): Money STILL NEEDED to finish the job.
       - Column headers: "Cost to Complete", "CTC", "Estimated Cost to Complete", "Remaining Costs"
       - This is how much more needs to be spent
       - Formula: Estimated Total Costs - Cost to Date = Cost to Complete
       - For completed jobs (100% done), this should be 0 or near 0

    3. ESTIMATED TOTAL COSTS (estimated_total_costs): Total expected cost when job is done.
       - Column headers: "Estimated Cost", "Total Est Cost", "Est Total Costs"
       - Formula: Cost to Date + Cost to Complete = Estimated Total Costs

    VALIDATION: For each row, verify: Cost to Date + Cost to Complete = Estimated Total Costs

    UNDER vs OVER BILLINGS:
    - UNDER BILLINGS (UB): Earned Revenue > Billed to Date (work done but not yet billed)
    - OVER BILLINGS (OB): Billed to Date > Earned Revenue (billed ahead of work)

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
    - Double-check Cost to Date vs Cost to Complete - they are different columns!
    """

    # Stage 2: LLM call
    raw_text = ""
    try:
        response = client.generate_content(
            prompt=prompt,
            model_name=state.model_name,
            pdf_bytes=file_bytes,
            response_mime_type="application/json",
            tracker=tracker,
            system_prompt="You are a financial document extraction engine specialized in construction Work-In-Progress schedules. Extract structured data and return it as a single JSON object. CRITICAL: Return ONLY the raw JSON object. No markdown code fences. No explanatory text before or after. No commentary.",
        )
        raw_text = response.text or ""
        _diag("llm_call", "OK", f"{len(raw_text)} chars returned")
    except Exception as e:
        _diag("llm_call", "FAIL", str(e))
        diagnostics["raw_response_preview"] = ""
        return {"processed_data": [], "totals_row": None, "metrics_data": tracker.get_metrics(), "extraction_diagnostics": diagnostics}

    # Stage 3: JSON parse
    diagnostics["raw_response_preview"] = raw_text[:500] if raw_text else "(empty)"
    parsed = None
    try:
        parsed = parse_json_safely(raw_text)
        if parsed is None:
            _diag("json_parse", "FAIL", "parse_json_safely returned None")
        else:
            _diag("json_parse", "OK", f"Type: {type(parsed).__name__}, "
                  + (f"Keys: {list(parsed.keys())}" if isinstance(parsed, dict) else f"Length: {len(parsed)}"))
    except Exception as e:
        _diag("json_parse", "FAIL", str(e))

    # Stage 3b: Normalize parsed JSON into {"rows": [...], "totals": {...}} structure
    data = None
    _row_field_keys = {"job_id", "total_contract_price", "cost_to_date"}

    if isinstance(parsed, dict):
        data = parsed
    elif isinstance(parsed, list) and len(parsed) > 0:
        # Bare array of rows: [{row1}, {row2}, ...]
        if isinstance(parsed[0], dict):
            data = {"rows": parsed}
            _diag("json_normalize", "OK", f"Wrapped top-level array ({len(parsed)} dicts) into rows")
        else:
            _diag("json_parse", "FAIL", f"Top-level list but items are {type(parsed[0]).__name__}, not dicts")

    if data is None:
        return {"processed_data": [], "totals_row": None, "metrics_data": tracker.get_metrics(), "extraction_diagnostics": diagnostics}

    # Stage 4: Row extraction with aggressive structural recovery
    raw_rows = []
    totals_data = data.get("totals")

    # Path A: Standard {"rows": [...]} structure
    candidate = data.get("rows")
    if isinstance(candidate, list) and len(candidate) > 0:
        raw_rows = candidate
        _diag("row_locate", "OK", f"Found {len(raw_rows)} rows under 'rows' key")

    # Path B: Alternative key names
    if not raw_rows:
        for alt_key in ("data", "jobs", "wip_rows", "schedule", "wip_schedule", "job_rows", "wip_data", "extracted_data"):
            candidate = data.get(alt_key)
            if isinstance(candidate, list) and len(candidate) > 0:
                raw_rows = candidate
                _diag("row_locate", "OK", f"Found {len(raw_rows)} rows under '{alt_key}' key")
                break

    # Path C: Scan all values — find any list of dicts that looks like rows
    if not raw_rows:
        for key, val in data.items():
            if key == "totals":
                continue
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                if _row_field_keys.issubset(set(val[0].keys())):
                    raw_rows = val
                    _diag("row_locate", "OK", f"Found {len(raw_rows)} row-like dicts under '{key}' key")
                    break

    # Path D: Top-level dict has row field names directly
    if not raw_rows and _row_field_keys.issubset(set(data.keys())):
        first_val = data.get("job_id")
        non_meta_data = {k: v for k, v in data.items() if k != "totals"}

        if isinstance(first_val, list):
            # Columnar format: {"job_id": ["101","102"], "cost_to_date": [1000, 2000]}
            num_rows = len(first_val)
            raw_rows = [
                {k: (v[i] if isinstance(v, list) and i < len(v) else v) for k, v in non_meta_data.items()}
                for i in range(num_rows)
            ]
            _diag("row_locate", "OK", f"Converted columnar format ({num_rows} rows)")
        else:
            # Single flat row (all fields at top level)
            raw_rows = [non_meta_data]
            _diag("row_locate", "OK", "Wrapped single flat object as 1 row")

    # Path E: Single nested wrapper — {"wip_schedule": {"rows": [...]}}
    if not raw_rows:
        for key, val in data.items():
            if key == "totals":
                continue
            if isinstance(val, dict):
                nested_rows = val.get("rows")
                if isinstance(nested_rows, list) and len(nested_rows) > 0:
                    raw_rows = nested_rows
                    totals_data = totals_data or val.get("totals")
                    _diag("row_locate", "OK", f"Found {len(raw_rows)} rows nested under '{key}.rows'")
                    break

    if not raw_rows:
        key_sample = {k: type(v).__name__ + (f"[{len(v)}]" if isinstance(v, (list, dict)) else "") for k, v in list(data.items())[:15]}
        _diag("row_parse", "FAIL", f"Could not locate rows in any known structure. Key types: {key_sample}")
        return {"processed_data": [], "totals_row": None, "metrics_data": tracker.get_metrics(), "extraction_diagnostics": diagnostics}

    # Stage 5: Parse individual rows
    rows: List[CalculatedWipRow] = []
    row_errors: List[Dict[str, Any]] = []
    for i, r in enumerate(raw_rows):
        try:
            rows.append(CalculatedWipRow(**r))
        except Exception as e:
            row_errors.append({"row_index": i, "error": str(e), "raw_keys": list(r.keys()) if isinstance(r, dict) else str(type(r))})

    if row_errors:
        _diag("row_parse", "PARTIAL" if rows else "FAIL",
               f"{len(rows)}/{len(raw_rows)} rows parsed, {len(row_errors)} failed")
        diagnostics["row_parse_errors"] = row_errors[:10]
    else:
        _diag("row_parse", "OK", f"{len(rows)} rows parsed")

    # Stage 6: Totals
    totals = None
    if totals_data:
        try:
            totals = WipTotals(**totals_data)
            _diag("totals_parse", "OK", "")
        except Exception as e:
            _diag("totals_parse", "FAIL", str(e))
    else:
        _diag("totals_parse", "SKIP", "No totals in response")

    return {"processed_data": rows, "totals_row": totals, "metrics_data": tracker.get_metrics(), "extraction_diagnostics": diagnostics}

# ==========================================
# 7. ANALYST NODE
# ==========================================

def analyst_node(state: WipState):
    print("--- RUNNING VALIDATIONS & ANALYSIS ---")

    try:
        rows = state.processed_data
        extracted_totals = state.totals_row

        if not rows:
            diag = state.extraction_diagnostics or {}
            stages = diag.get("stages", [])

            # Build human-readable failure chain
            failure_chain = []
            last_ok_stage = "none"
            failed_stage = "unknown"
            for s in stages:
                if s["status"] == "OK":
                    last_ok_stage = s["stage"]
                else:
                    failure_chain.append(f"{s['stage']}: {s['detail']}")
                    if failed_stage == "unknown":
                        failed_stage = s["stage"]

            raw_preview = diag.get("raw_response_preview", "(not captured)")
            row_parse_errors = diag.get("row_parse_errors", [])

            # Determine specific structural message based on where it failed
            if failed_stage == "file_read":
                struct_msg = f"File Read Failed: {failure_chain[0] if failure_chain else 'unknown error'}"
            elif failed_stage == "llm_call":
                struct_msg = f"Model Call Failed: {failure_chain[0] if failure_chain else 'API error'}"
            elif failed_stage == "json_parse":
                # Show what the model actually returned
                preview = raw_preview[:200].replace('\n', ' ').strip()
                struct_msg = f"JSON Parse Failed — model returned: \"{preview}...\""
            elif failed_stage in ("row_locate", "row_parse"):
                # JSON was valid but structure didn't match or rows couldn't parse
                last_detail = failure_chain[-1] if failure_chain else "unknown structure"
                struct_msg = f"Data structure issue: {last_detail}"
            else:
                struct_msg = f"Extraction failed at: {failed_stage}"

            # Formulaic message should explain what we know
            if row_parse_errors:
                formulaic_msg = f"{len(row_parse_errors)} rows failed to parse: {row_parse_errors[0].get('error', '')[:100]}"
                formulaic_details = [{"id": f"row_{e['row_index']}", "msg": e["error"][:120]} for e in row_parse_errors[:5]]
            else:
                formulaic_msg = f"No data to validate (extraction failed at {failed_stage} stage)"
                formulaic_details = []

            return {
                "final_json": {"error": "No data extracted", "extraction_diagnostics": diag},
                "validation_errors": [],
                "correction_suggestions": [],
                "correction_log": [],
                "surety_risk_context": {},
                "risk_rows": [],
                "widget_data": {
                    "validations": {
                        "structural": {
                            "passed": False,
                            "message": struct_msg,
                            "details": [{"id": "DIAG", "msg": c} for c in failure_chain],
                        },
                        "formulaic": {
                            "passed": False,
                            "message": formulaic_msg,
                            "details": formulaic_details,
                            "jobIssues": [],
                        },
                        "totals": {"passed": False, "message": "No data to validate", "details": []},
                    },
                    "extraction_failure": {
                        "summary": " → ".join(failure_chain) if failure_chain else "Unknown failure",
                        "failed_stage": failed_stage,
                        "last_ok_stage": last_ok_stage,
                        "stages": stages,
                        "raw_preview": raw_preview[:500],
                        "row_parse_errors": row_parse_errors[:5],
                    },
                    "metrics": {
                        "row1_1": {"label": "Contract Value", "value": "—"},
                        "row1_2": {"label": "UEGP", "value": "—"},
                        "row1_3": {"label": "CTC", "value": "—"},
                        "row2_1": {"label": "Earned Rev", "value": "—"},
                        "row2_2": {"label": "GP %", "value": "—"},
                        "row2_3": {"label": "Net UB / OB", "value": "—"},
                    },
                    "riskTier": "UNKNOWN",
                    "riskRowsAll": [],
                },
            }

        validations = build_validations()

        # ---- PHASE 1: Initial UB/OB calc + validation (pre-correction) ----
        for r in rows:
            variance = r.revenues_earned - r.billed_to_date
            if variance > 0:
                r.under_billings_calc = variance
                r.over_billings_calc = 0.0
            else:
                r.under_billings_calc = 0.0
                r.over_billings_calc = abs(variance)

        pre_correction_errors: List[Dict[str, Any]] = []
        all_correction_suggestions: List[Dict[str, Any]] = []

        for r in rows:
            row_errors = run_validations(r, validations)
            pre_correction_errors.extend(row_errors)
            if row_errors:
                suggestions = suggest_corrections(r, row_errors)
                all_correction_suggestions.extend(suggestions)

        # ---- PHASE 2: Detect column swaps ----
        column_swaps = detect_column_swaps(rows)

        # ---- PHASE 3: Apply corrections ----
        correction_log = apply_corrections(rows, column_swaps, all_correction_suggestions)

        # ---- PHASE 4: Recompute UB/OB on corrected data ----
        for r in rows:
            variance = r.revenues_earned - r.billed_to_date
            if variance > 0:
                r.under_billings_calc = variance
                r.over_billings_calc = 0.0
            else:
                r.under_billings_calc = 0.0
                r.over_billings_calc = abs(variance)

        # ---- PHASE 5: Re-validate post-correction ----
        post_correction_errors: List[Dict[str, Any]] = []
        for r in rows:
            row_errors = run_validations(r, validations)
            post_correction_errors.extend(row_errors)

        # Use post-correction errors as the "final" validation state
        all_validation_errors = post_correction_errors

        # ---- PHASE 6: Aggregate totals from corrected data ----
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
            calc.under_billings += r.under_billings_calc
            calc.over_billings += r.over_billings_calc
            calc.uegp += r.uegp

        # ---- PHASE 7: Full totals validation (all columns) ----
        _totals_fields = [
            ("total_contract_price", "Contract Price"),
            ("estimated_total_costs", "Est Total Costs"),
            ("estimated_gross_profit", "Est Gross Profit"),
            ("revenues_earned", "Revenues Earned"),
            ("cost_to_date", "Cost to Date"),
            ("gross_profit_to_date", "GP to Date"),
            ("billed_to_date", "Billed to Date"),
            ("cost_to_complete", "Cost to Complete"),
        ]

        totals_pass = False
        totals_msg = "No Totals Row"
        totals_details: List[Dict[str, Any]] = []

        if extracted_totals:
            mismatches = []
            for field_name, display_name in _totals_fields:
                calc_val = getattr(calc, field_name, 0.0)
                ext_val = getattr(extracted_totals, field_name, 0.0)
                diff = abs(calc_val - ext_val)
                if diff >= 1000.0:
                    mismatches.append((field_name, display_name, calc_val, ext_val, diff))

            if not mismatches:
                totals_pass = True
                totals_msg = f"All {len(_totals_fields)} column sums match report totals"
            else:
                passed_count = len(_totals_fields) - len(mismatches)
                totals_msg = f"{passed_count}/{len(_totals_fields)} columns match ({len(mismatches)} mismatches)"
                for fname, dname, cv, ev, diff in mismatches:
                    totals_details.append({
                        "id": "TOTALS",
                        "field": fname,
                        "msg": f"{dname}: Calc ${cv:,.0f} vs Report ${ev:,.0f} (diff ${diff:,.0f})",
                    })

        # ---- PHASE 8: Risk analysis on corrected data ----
        surety_risk_context = build_surety_risk_context(rows, calc)
        surety_risk_context["risk_tier"] = compute_portfolio_risk_tier(surety_risk_context)

        risk_rows = build_risk_rows(rows, calc)

        # ---- Build widget data ----
        t_uegp = calc.estimated_gross_profit - calc.gross_profit_to_date
        gp_percent = (calc.gross_profit_to_date / calc.revenues_earned * 100) if calc.revenues_earned else 0

        net_ub_ob = calc.under_billings - calc.over_billings
        net_ub_ob_label = f"Under ${net_ub_ob/1000:.0f}k" if net_ub_ob >= 0 else f"Over ${abs(net_ub_ob)/1000:.0f}k"

        struct_pass = len(rows) > 0 and all(r.job_id for r in rows)
        struct_msg = "Structure Valid" if struct_pass else "Missing IDs/Data"

        errors_by_job: Dict[str, List[Dict[str, Any]]] = {}
        for e in all_validation_errors:
            errors_by_job.setdefault(e["job_id"], []).append(e)

        formula_errors_display = [{"id": e["job_id"], "msg": e["message"]} for e in all_validation_errors]

        formula_pass = len(all_validation_errors) == 0
        if formula_pass and correction_log:
            formula_msg = f"Column Math Validated (after {len(correction_log)} auto-corrections)"
        elif formula_pass:
            formula_msg = "Column Math Validated"
        else:
            formula_msg = f"Column Math Issues ({len(errors_by_job)} rows)"

        # Build per-job issue display (corrections + remaining errors)
        jobs_with_issues: Dict[str, Dict[str, Any]] = {}
        for e in all_validation_errors:
            job_id = e["job_id"]
            jobs_with_issues.setdefault(job_id, {"errors": [], "corrections": []})
            jobs_with_issues[job_id]["errors"].append(
                {"validation": e["validation"], "message": e["message"], "category": e["category"]}
            )

        for s in all_correction_suggestions:
            job_id = s["job_id"]
            jobs_with_issues.setdefault(job_id, {"errors": [], "corrections": []})
            applied = any(
                cl["job_id"] == job_id and cl.get("field") == s.get("field")
                for cl in correction_log if cl["type"] == "digit_fix"
            )
            jobs_with_issues[job_id]["corrections"].append(
                {
                    "field": s["field"],
                    "current": f"${s['current_value']:,.0f}",
                    "suggested": f"${s['suggested_value']:,.0f}",
                    "confidence": s["confidence"],
                    "cross_validation_count": s.get("cross_validation_count", 1),
                    "applied": applied,
                    "reasoning": s["reasoning"],
                }
            )

        # Include column swap info in job issues
        for swap in column_swaps:
            job_id = swap["job_id"]
            jobs_with_issues.setdefault(job_id, {"errors": [], "corrections": []})
            jobs_with_issues[job_id]["corrections"].append(
                {
                    "field": "cost_to_date ↔ cost_to_complete",
                    "current": f"CTD ${swap['original_ctd']:,.0f} / CTC ${swap['original_ctc']:,.0f}",
                    "suggested": f"CTD ${swap['original_ctc']:,.0f} / CTC ${swap['original_ctd']:,.0f}",
                    "confidence": "high",
                    "cross_validation_count": 0,
                    "applied": True,
                    "reasoning": swap["reasoning"],
                }
            )

        corrections_display = [{"jobId": job_id, **data} for job_id, data in jobs_with_issues.items()]

        widget_data = {
            "validations": {
                "structural": {"passed": struct_pass, "message": struct_msg, "details": []},
                "formulaic": {"passed": formula_pass, "message": formula_msg, "details": formula_errors_display, "jobIssues": corrections_display},
                "totals": {"passed": totals_pass, "message": totals_msg, "details": totals_details},
            },
            "corrections_applied": correction_log,
            "metrics": {
                "row1_1": {"label": "Contract Value", "value": f"${calc.total_contract_price/1000000:.2f}M"},
                "row1_2": {"label": "UEGP", "value": f"${t_uegp/1000000:.2f}M"},
                "row1_3": {"label": "CTC", "value": f"${calc.cost_to_complete/1000000:.2f}M"},
                "row2_1": {"label": "Earned Rev", "value": f"${calc.revenues_earned/1000000:.2f}M"},
                "row2_2": {"label": "GP %", "value": f"{gp_percent:.1f}%"},
                "row2_3": {"label": "Net UB / OB", "value": net_ub_ob_label},
            },
            "riskTier": surety_risk_context["risk_tier"],
            "riskRowsAll": risk_rows,
        }

        return {
            "calculated_totals": calc,
            "validation_errors": all_validation_errors,
            "correction_suggestions": all_correction_suggestions,
            "correction_log": correction_log,
            "surety_risk_context": surety_risk_context,
            "risk_rows": risk_rows,
            "widget_data": widget_data,
        }

    except Exception as e:
        print(f"ANALYST NODE ERROR: {e}")
        print(traceback.format_exc())
        return {
            "calculated_totals": WipTotals(),
            "validation_errors": [],
            "correction_suggestions": [],
            "correction_log": [],
            "surety_risk_context": {"risk_tier": "UNKNOWN", "error": str(e)},
            "risk_rows": [],
            "widget_data": {
                "validations": {
                    "structural": {"passed": False, "message": f"Error: {str(e)}", "details": []},
                    "formulaic": {"passed": False, "message": "Error during analysis", "details": [], "jobIssues": []},
                    "totals": {"passed": False, "message": "Error during analysis", "details": []},
                },
                "metrics": {
                    "row1_1": {"label": "Contract Value", "value": "$0.00M"},
                    "row1_2": {"label": "UEGP", "value": "$0.00M"},
                    "row1_3": {"label": "CTC", "value": "$0.00M"},
                    "row2_1": {"label": "Earned Rev", "value": "$0.00M"},
                    "row2_2": {"label": "GP %", "value": "0.0%"},
                    "row2_3": {"label": "Net UB / OB", "value": "Error"},
                },
                "riskTier": "UNKNOWN",
                "riskRowsAll": [],
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            "final_json": {"error": str(e), "traceback": traceback.format_exc()},
        }

# ==========================================
# 8. NARRATIVE NODE (metrics via from_dict, no bare except)
# ==========================================

SURETY_NARRATIVE_PROMPT = """
You are a senior surety underwriting analyst preparing an executive summary for the chief underwriter. This summary accompanies a WIP schedule review and should demonstrate the caliber of analysis expected at a top-10 surety.

SURETY PERSPECTIVE: We have bonded this contractor. Our exposure is the penal sum of outstanding bonds. If the contractor defaults mid-project, we inherit the obligation to complete or pay. Every data point should be evaluated through this lens: "What does this mean for our recovery position and likelihood of claim?"

ANALYTICAL FRAMEWORK (priority order):

1. CASH FLOW & LIQUIDITY SIGNAL
   - Underbilling = work performed but unbilled = cash NOT collected. This is the single biggest red flag because it directly reduces recovery in a default scenario. Quantify the exposure.
   - Overbilling = cash collected ahead of work = favorable for recovery (but can mask front-loading or manipulation). Only flag if extreme or suspicious.
   - Net billing position relative to portfolio size tells the real story.

2. LOSS EXPOSURE & MARGIN INTEGRITY
   - Loss jobs represent direct erosion of the contractor's net worth and bonding capacity.
   - GP fade (actual profit lagging expected profit at current completion) often precedes loss recognition — it means the contractor's estimates may be stale.
   - Thin margins (<5% GP) leave no buffer. One change order or weather delay flips these to losses.

3. COMPLETION & CONCENTRATION
   - Early-stage large jobs have the most remaining exposure. Late-stage jobs are de-risking.
   - Single-job concentration means one default cascades. Diversified portfolios absorb hits.

4. DATA QUALITY CONTEXT
   {correction_context}

WHAT TO WRITE:
- 4-6 sentences as flowing prose (no bullets, no headers, no bold).
- Open with the single most important finding — the one thing the chief underwriter needs to know first.
- Connect signals to conclusions: don't just report numbers, explain what they MEAN for our bond program.
- Distinguish between systemic concerns (patterns across jobs) vs isolated issues (one bad job).
- If the portfolio is genuinely healthy, say so with conviction and explain the structural reasons why (e.g., "margins are thick across the board with balanced billing positions").
- End with a forward-looking statement: what to watch at next review, or what additional information would sharpen the picture.
- DO NOT repeat stats already displayed in the dashboard (job count, total contract value, risk tier label). The underwriter can see those. Add insight they can't get from the numbers alone.

RISK DATA:
{risk_context}

SUMMARY:"""


def narrative_node(state: WipState):
    print("--- GENERATING NARRATIVE ---")
    print(f"--- USING MODEL: {state.model_name} ---")

    tracker = MetricsTracker.from_dict(state.metrics_data, state.model_name)
    client = get_client()

    try:
        risk_context = state.surety_risk_context

        if not risk_context:
            widget_data = dict(state.widget_data or {})
            widget_data["summary"] = {"text": "Unable to generate narrative: no risk context available."}
            return {"narrative": widget_data["summary"]["text"], "widget_data": widget_data, "metrics_data": tracker.get_metrics()}

        if "error" in risk_context:
            widget_data = dict(state.widget_data or {})
            widget_data["summary"] = {"text": f"Analysis error: {risk_context.get('error')}"}
            return {"narrative": widget_data["summary"]["text"], "widget_data": widget_data, "metrics_data": tracker.get_metrics()}

        # Build correction context for the narrative
        correction_log = state.correction_log or []
        correction_suggestions = state.correction_suggestions or []
        validation_errors = state.validation_errors or []

        if not correction_log and not validation_errors:
            correction_context = "All extraction checks passed. Data quality is high confidence."
        else:
            parts = []
            if correction_log:
                parts.append(f"{len(correction_log)} auto-corrections were applied (column swaps or digit fixes).")
            if validation_errors:
                parts.append(f"{len(validation_errors)} validation issues remain post-correction.")
            if correction_suggestions:
                low_conf = [s for s in correction_suggestions if s.get("confidence") != "high"]
                if low_conf:
                    parts.append(f"{len(low_conf)} suggested fixes were not applied (low confidence).")
            correction_context = " ".join(parts) + " Factor data quality into your confidence level."

        prompt = SURETY_NARRATIVE_PROMPT.format(
            risk_context=json.dumps(risk_context, indent=2),
            correction_context=correction_context,
        )

        response = client.generate_content(
            prompt=prompt,
            model_name=state.model_name,
            temperature=0.3,
            max_tokens=400,
            tracker=tracker,
            system_prompt=(
                "You are a senior surety underwriting analyst at a top-10 surety company. "
                "Write with precision, authority, and analytical depth. "
                "Prose only — no bullets, no headers, no formatting. "
                "Every sentence should earn its place."
            ),
        )

        narrative = response.text.strip()
        widget_data = dict(state.widget_data or {})
        widget_data["summary"] = {"text": narrative}

        return {"narrative": narrative, "widget_data": widget_data, "metrics_data": tracker.get_metrics()}

    except Exception as e:
        print(f"NARRATIVE NODE ERROR: {e}")
        print(traceback.format_exc())

        fallback = "Error generating narrative summary."
        try:
            rc = state.surety_risk_context or {}
            portfolio = rc.get("portfolio", {})
            total_value = portfolio.get("total_contract_value", 0) or 0
            fallback = (
                f"Portfolio contains {portfolio.get('total_jobs', 0)} jobs with "
                f"${total_value/1000000:.1f}M total contract value. "
                f"Risk tier: {rc.get('risk_tier', 'Unknown')}."
            )
        except Exception as _e:
            logger.debug("Narrative fallback construction failed: %s", _e, exc_info=True)

        widget_data = dict(state.widget_data or {})
        widget_data["summary"] = {"text": fallback}
        widget_data["narrative_error"] = str(e)
        return {"narrative": fallback, "widget_data": widget_data, "metrics_data": tracker.get_metrics()}

# ==========================================
# 9. OUTPUT NODE (propagate error)
# ==========================================

def output_node(state: WipState):
    print("--- BUILDING FINAL OUTPUT ---")

    try:
        payload = {
            "clean_table": [r.model_dump(exclude={"under_billings_calc", "over_billings_calc"}) for r in state.processed_data] if state.processed_data else [],
            "calculated_totals": state.calculated_totals.model_dump() if state.calculated_totals else {},
            "validation_errors": state.validation_errors or [],
            "correction_suggestions": state.correction_suggestions or [],
            "corrections_applied": state.correction_log or [],
            "surety_risk_context": state.surety_risk_context or {},
            "widget_data": state.widget_data or {},
            "metrics": state.metrics_data or {},
        }

        # If upstream set an error, surface it
        if state.final_json and "error" in state.final_json:
            payload["error"] = state.final_json.get("error")
            if "traceback" in state.final_json:
                payload["traceback"] = state.final_json.get("traceback")

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
                "widget_data": {"error": str(e)},
                "metrics": state.metrics_data or {},
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
