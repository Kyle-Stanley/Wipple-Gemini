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
    OPTIMAL_MIX_KEY,
    OPTIMAL_MIX_PRIMARY,
    OPTIMAL_MIX_FALLBACK,
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
    extraction_feedback: str = ""  # Passed on retry runs; empty on first pass

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
    """Generate correction suggestions ONLY for clear OCR/extraction errors."""
    all_candidates: List[tuple] = []  # (validation_name, candidate_dict)
    for error in errors:
        candidates = _get_candidates_for_validation(row, error["validation"])
        for c in candidates:
            all_candidates.append((error["validation"], c))

    field_validation_count: Dict[str, int] = {}
    seen: Dict[str, set] = {}
    for vname, c in all_candidates:
        seen.setdefault(c["field"], set()).add(vname)
    field_validation_count = {f: len(v) for f, v in seen.items()}

    suggestions: List[Dict[str, Any]] = []

    for error in errors:
        candidates = _get_candidates_for_validation(row, error["validation"])
        if not candidates:
            continue

        scored = []
        for c in candidates:
            base_score = digit_change_score(c["current"], c["suggested"])
            cross_count = field_validation_count.get(c["field"], 1)
            effective_score = max(0, base_score - (cross_count - 1))
            scored.append((effective_score, base_score, c, cross_count))

        scored.sort(key=lambda x: x[0])
        effective, raw, best, cross_count = scored[0]

        if raw <= 2:
            confidence = "high" if effective <= 1 else "medium"
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
                        f"Likely OCR misread: {best['field']} ${best['current']:,.0f} → ${best['suggested']:,.0f} "
                        f"({raw} digit change"
                        + (f", {cross_count} validations agree" if cross_count > 1 else "")
                        + f") fixes: {best['formula']}"
                    ),
                }
            )
        else:
            suggestions.append(
                {
                    "job_id": row.job_id,
                    "field": best["field"],
                    "current_value": best["current"],
                    "suggested_value": None,
                    "confidence": "flag",
                    "digit_changes": raw,
                    "effective_score": effective,
                    "cross_validation_count": cross_count,
                    "reasoning": (
                        f"Validation failed on {best['formula']} but fix requires {raw} digit changes — "
                        f"likely a column misread rather than OCR error. Manual review recommended."
                    ),
                }
            )

    return suggestions


# ==========================================
# 4b. ROOT CAUSE RESOLVER
# ==========================================

def resolve_root_causes(
    row: CalculatedWipRow,
    errors: List[Dict[str, Any]],
    validations: List[Validation],
) -> Dict[str, Any]:
    """
    For a single row with N failing validations, simulate substituting each candidate
    field value and measure how many failures resolve. Uses exclusive validation failure
    counting to disambiguate tied fields (e.g. Est Cost == CTD) without name heuristics.
    Returns root cause tags per validation.
    """
    if not errors:
        return {"by_validation": {}, "root_causes": []}

    failing_names = {e["validation"] for e in errors}

    # Collect unique (field -> best_suggested_value) across all failing validations
    all_candidates: Dict[str, float] = {}
    candidate_sources: Dict[str, List[str]] = {}

    for error in errors:
        for c in _get_candidates_for_validation(row, error["validation"]):
            field = c["field"]
            suggested = c["suggested"]
            if field not in all_candidates:
                all_candidates[field] = suggested
                candidate_sources[field] = [error["validation"]]
            else:
                existing_score = digit_change_score(getattr(row, field, 0), all_candidates[field])
                new_score = digit_change_score(getattr(row, field, 0), suggested)
                if new_score < existing_score:
                    all_candidates[field] = suggested
                candidate_sources[field].append(error["validation"])

    # Simulate each substitution and count resolutions
    resolution_counts: Dict[str, int] = {}
    for field, suggested_value in all_candidates.items():
        sim_row = row.model_copy()
        setattr(sim_row, field, suggested_value)
        sim_errors = {e["validation"] for e in run_validations(sim_row, validations)}
        resolution_counts[field] = len(failing_names - sim_errors)

    if not resolution_counts:
        return {"by_validation": {}, "root_causes": []}

    max_resolutions = max(resolution_counts.values())
    root_fields = {f for f, c in resolution_counts.items() if c == max_resolutions and c > 0}

    # Tie-break using exclusive validation failures.
    # When multiple fields resolve the same number of failures (e.g. Est Cost == CTD),
    # count how many FAILING validations each tied field appears in EXCLUSIVELY —
    # i.e. the other tied fields are NOT in that validation's fields_involved list.
    # The field with more exclusive failures is the actual bad column.
    # This uses the algebraic structure of the validation system itself rather than
    # any name-based heuristic, so it works regardless of which values happen to be equal.
    if len(root_fields) > 1:
        failing_validation_names = {e["validation"] for e in errors}
        exclusive_failure_counts: Dict[str, int] = {f: 0 for f in root_fields}

        for v in validations:
            if v.name not in failing_validation_names:
                continue
            tied_fields_in_v = [f for f in root_fields if f in v.fields_involved]
            if len(tied_fields_in_v) == 1:
                # This validation is exclusive to one of the tied fields
                exclusive_failure_counts[tied_fields_in_v[0]] += 1

        max_exclusive = max(exclusive_failure_counts.values())
        if max_exclusive > 0:
            # At least one field has exclusive failures — keep only those
            root_fields = {f for f, c in exclusive_failure_counts.items() if c == max_exclusive}
        # If all tied fields have 0 exclusive failures (every failing validation
        # involves all of them), we cannot disambiguate algebraically.
        # In that case, keep all root_fields — the frontend will show them all as suspect.

    # Build shadowed set: fields sharing the exact same extracted value as a confirmed
    # root field are the same bad column read twice — suppress them from per-validation tags.
    confirmed_root_values = {getattr(row, f, None) for f in root_fields}
    shadowed_fields: set = set()
    for field in list(resolution_counts.keys()):
        if field in root_fields:
            continue
        if getattr(row, field, None) in confirmed_root_values:
            shadowed_fields.add(field)

    by_validation: Dict[str, Dict] = {}
    for error in errors:
        vname = error["validation"]
        v_candidates = _get_candidates_for_validation(row, vname)
        eligible = [c for c in v_candidates if c["field"] not in shadowed_fields]
        root_candidate = next((c for c in eligible if c["field"] in root_fields), None)
        if root_candidate:
            field = root_candidate["field"]
            by_validation[vname] = {
                "is_cascade": False,
                "root_cause_field": field,
                "root_cause_value": all_candidates[field],
                "resolutions": resolution_counts[field],
            }
        else:
            best_field = max(
                (f for f in resolution_counts if f not in shadowed_fields),
                key=resolution_counts.get,
                default=max(resolution_counts, key=resolution_counts.get),
            )
            by_validation[vname] = {
                "is_cascade": True,
                "root_cause_field": best_field,
                "root_cause_value": all_candidates.get(best_field),
                "resolutions": resolution_counts.get(best_field, 0),
            }

    return {
        "by_validation": by_validation,
        "root_causes": [
            {
                "field": field,
                "suggested_value": all_candidates[field],
                "resolutions": resolution_counts[field],
                "nominated_by": candidate_sources[field],
            }
            for field in root_fields
        ],
    }


def detect_portfolio_column_errors(
    rows: List[CalculatedWipRow],
    all_root_cause_results: List[Dict[str, Any]],
    total_error_count: int,
    validations: List[Validation],
    threshold: float = 0.90,
    min_error_rows: int = 5,
) -> List[Dict[str, Any]]:
    """
    Aggregate root cause data across all rows. A column error is declared when a
    single field substitution resolves >= threshold% of the ROWS that have errors
    (not total error count), and at least min_error_rows rows have errors.

    Using row-based thresholding prevents a single bad job on a clean document from
    triggering a portfolio-wide column correction. Digit-score is intentionally not
    considered here — column errors involve grabbing the wrong column entirely, so
    values have no digit relationship to the correct ones.

    Returns a list of detected column errors (usually 0 or 1).
    """
    if total_error_count == 0:
        return []

    # How many distinct rows have at least one error?
    rows_with_errors: set = {rc["job_id"] for rc in all_root_cause_results if rc.get("root_causes")}
    if len(rows_with_errors) < min_error_rows:
        return []

    total_rows = len(rows)

    # For each field, count how many error-rows are fully resolved by substituting it
    field_rows_resolved: Dict[str, set] = {}   # field -> set of job_ids fully resolved
    field_values: Dict[str, Dict[str, float]] = {}  # field -> {job_id -> suggested_value}

    for rc in all_root_cause_results:
        job_id = rc["job_id"]
        job_error_count = len([e for e in rc.get("by_validation", {}).values()])
        for cause in rc.get("root_causes", []):
            field = cause["field"]
            # A row is "resolved" by this field if it fixes ALL of that row's errors
            if cause["resolutions"] >= job_error_count and job_error_count > 0:
                field_rows_resolved.setdefault(field, set()).add(job_id)
            field_values.setdefault(field, {})[job_id] = cause["suggested_value"]

    column_errors = []
    for field, resolved_job_ids in field_rows_resolved.items():
        # Threshold over total rows — a true column error affects every job
        row_resolution_rate = len(resolved_job_ids) / total_rows
        if row_resolution_rate >= threshold:
            # Apply correction to all affected rows
            rows_corrected = 0
            for r in rows:
                job_suggested = field_values.get(field, {}).get(r.job_id)
                if job_suggested is not None:
                    setattr(r, field, job_suggested)
                    rows_corrected += 1

            column_errors.append({
                "field": field,
                "row_resolution_rate": row_resolution_rate,
                "rows_resolved": len(resolved_job_ids),
                "total_rows": total_rows,
                "rows_with_errors": len(rows_with_errors),
                "rows_corrected": rows_corrected,
            })

    return column_errors


# ==========================================
# 4c. COLUMN SWAP DETECTION (Tier 1)
# ==========================================

def detect_column_swaps(rows: List[CalculatedWipRow]) -> List[Dict[str, Any]]:
    """Detect likely Cost to Date <-> Cost to Complete column swaps."""
    swaps: List[Dict[str, Any]] = []
    for r in rows:
        if r.estimated_total_costs <= 0 or r.total_contract_price <= 0:
            continue
        if r.cost_to_date <= 0 and r.cost_to_complete <= 0:
            continue

        revenue_poc = r.revenues_earned / r.total_contract_price
        cost_poc = r.cost_to_date / r.estimated_total_costs

        if (revenue_poc > 0.5 and cost_poc < 0.5
                and r.cost_to_complete > r.cost_to_date
                and r.cost_to_complete > 0):

            swapped_poc = r.cost_to_complete / r.estimated_total_costs
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
    """Apply high-confidence corrections in-place and return a log."""
    correction_log: List[Dict[str, Any]] = []

    swap_jobs = {s["job_id"] for s in column_swaps}

    digit_fixes: Dict[str, List[Dict]] = {}
    for d in digit_corrections:
        if d.get("confidence") == "high" and d.get("suggested_value") is not None:
            digit_fixes.setdefault(d["job_id"], []).append(d)

    for r in rows:
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
# 5. SURETY RISK ANALYSIS
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

    early_stage_jobs = []
    for r in rows:
        if r.estimated_total_costs > 0 and r.total_contract_price > 500000:
            poc = r.cost_to_date / r.estimated_total_costs
            remaining = r.estimated_total_costs - r.cost_to_date
            if poc < 0.25 and remaining > 100000:
                early_stage_jobs.append({
                    "id": r.job_id, "name": r.job_name,
                    "contract": r.total_contract_price,
                    "pct_complete": poc,
                    "remaining_cost": remaining,
                })

    billing_lag_jobs = []
    for r in rows:
        if r.cost_to_date > 50000 and r.billed_to_date > 0:
            ratio = r.billed_to_date / r.cost_to_date
            if ratio < 0.80:
                billing_lag_jobs.append({
                    "id": r.job_id, "name": r.job_name,
                    "cost_to_date": r.cost_to_date,
                    "billed_to_date": r.billed_to_date,
                    "billing_ratio": ratio,
                    "cash_gap": r.cost_to_date - r.billed_to_date,
                })

    aggregate_poc = calc.cost_to_date / calc.estimated_total_costs if calc.estimated_total_costs else 0
    total_remaining_cost = calc.estimated_total_costs - calc.cost_to_date
    jobs_over_90 = sum(1 for r in rows if r.estimated_total_costs > 0 and r.cost_to_date / r.estimated_total_costs > 0.90)
    jobs_under_25 = sum(1 for r in rows if r.estimated_total_costs > 0 and r.cost_to_date / r.estimated_total_costs < 0.25)

    ub_ob_mismatch_jobs = [r for r in rows if r.ub_ob_discrepancy_abs > 100]

    return {
        "portfolio": {
            "total_jobs": len(rows),
            "total_contract_value": calc.total_contract_price,
            "aggregate_poc": aggregate_poc,
            "net_billing_position": calc.under_billings - calc.over_billings,
            "total_uegp": total_uegp,
            "total_gp_margin": calc.estimated_gross_profit / calc.total_contract_price if calc.total_contract_price else 0,
            "total_remaining_cost": total_remaining_cost,
            "jobs_over_90_pct": jobs_over_90,
            "jobs_under_25_pct": jobs_under_25,
        },
        "cash_risk": {
            "severe_ub_count": len(severe_ub_jobs),
            "severe_ub_jobs": [
                {"id": r.job_id, "name": r.job_name, "ub_amount": r.under_billings_calc, "contract": r.total_contract_price}
                for r in severe_ub_jobs
            ],
            "total_ub_exposure": total_ub_exposure,
            "billing_lag_count": len(billing_lag_jobs),
            "billing_lag_jobs": billing_lag_jobs[:5],
            "total_billing_gap": sum(j["cash_gap"] for j in billing_lag_jobs),
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
        "exposure_risk": {
            "early_stage_count": len(early_stage_jobs),
            "early_stage_jobs": early_stage_jobs[:5],
            "total_early_stage_remaining": sum(j["remaining_cost"] for j in early_stage_jobs),
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

    if risk_context["cash_risk"]["billing_lag_count"] >= 3:
        score += 2
    elif risk_context["cash_risk"]["billing_lag_count"] >= 1:
        score += 1

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

    if risk_context["exposure_risk"]["early_stage_count"] >= 2:
        score += 2
    elif risk_context["exposure_risk"]["early_stage_count"] >= 1:
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

        if r.estimated_total_costs > 0:
            poc = r.cost_to_date / r.estimated_total_costs
            remaining_cost = r.estimated_total_costs - r.cost_to_date
            if poc < 0.25 and r.total_contract_price > 500000 and remaining_cost > 100000:
                job_risk_tags.append("Early Stage Exposure")
                risk_score += 12 + remaining_cost / 100000
                risk_details.append(
                    {
                        "tag": "Early Stage Exposure",
                        "summary": f"Only {poc:.0%} complete, ${remaining_cost:,.0f} in costs remaining",
                        "detail": "Large job in early stages means maximum remaining exposure. Most of the risk is still ahead — cost overruns, disputes, and schedule delays are most likely to emerge as work ramps up.",
                    }
                )

        if r.total_contract_price > 0 and r.estimated_total_costs > 0:
            cost_to_contract = r.estimated_total_costs / r.total_contract_price
            if cost_to_contract > 0.97 and r.estimated_gross_profit >= 0:
                margin_pct = (r.estimated_gross_profit / r.total_contract_price * 100)
                job_risk_tags.append("Cost Overrun Signal")
                risk_score += 18
                risk_details.append(
                    {
                        "tag": "Cost Overrun Signal",
                        "summary": f"Costs at {cost_to_contract:.0%} of contract, only {margin_pct:.1f}% margin remaining",
                        "detail": "Estimated costs have nearly consumed the entire contract value. Any additional cost growth flips this to a loss job. Check for pending change orders that might restore margin.",
                    }
                )

        if r.cost_to_date > 50000 and r.billed_to_date > 0:
            billing_ratio = r.billed_to_date / r.cost_to_date
            if billing_ratio < 0.80:
                cash_gap = r.cost_to_date - r.billed_to_date
                job_risk_tags.append("Billing Lag")
                risk_score += 8 + cash_gap / 50000
                risk_details.append(
                    {
                        "tag": "Billing Lag",
                        "summary": f"Billed only {billing_ratio:.0%} of costs incurred (${cash_gap:,.0f} gap)",
                        "detail": "Cash collections are significantly behind costs spent. This creates cash flow pressure — the contractor is funding this job out of pocket or from other jobs' cash. Sustained billing lag across multiple jobs signals liquidity stress.",
                    }
                )

        if r.estimated_total_costs > 0:
            poc_check = r.cost_to_date / r.estimated_total_costs
            if poc_check > 0.95 and r.cost_to_complete > r.total_contract_price * 0.05 and r.cost_to_complete > 25000:
                job_risk_tags.append("Stale CTC")
                risk_score += 5
                risk_details.append(
                    {
                        "tag": "Stale CTC",
                        "summary": f"{poc_check:.0%} complete but ${r.cost_to_complete:,.0f} CTC remaining",
                        "detail": "Job appears nearly complete by percentage but still carries meaningful cost to complete. May indicate punch list issues, retainage disputes, or stale estimates that haven't been updated.",
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
                    "underBillings": ub,
                    "overBillings": ob,
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
    Extract all job rows and the TOTALS row from this WIP schedule.

    BEFORE EXTRACTING ANY ROWS:
    1. Scan the entire header row(s) and number every column left to right (1, 2, 3...)
    2. Column headers may span multiple lines — treat wrapped text as a single header
    3. Explicitly assign each column number to one of the 10 field names below
    4. Only after the full column map is established, extract every row using those fixed positions
    5. Apply the exact same column mapping to row 1 as to every other row

    TYPICAL COLUMN ORDER (left to right):
    Job ID → Job Name → Contract Price → Estimated Total Costs → Estimated Gross Profit → Revenues Earned → Cost to Date → Gross Profit to Date → Billed to Date → Cost to Complete → Under Billings → Over Billings

    Use this order as a positional tiebreaker when column headers are ambiguous or truncated.

    COLUMN IDENTIFICATION — all 10 fields:

    1. total_contract_price — The full agreed contract value.
       Headers: "Contract", "Contract Price", "Contract Amount", "Total Contract", "Revised Contract"

    2. estimated_total_costs — The projected total cost to complete the entire job.
       Headers: "Estimated Cost", "Est Total Cost", "Total Est Cost", "Revised Est Cost",
                "Total Projected Costs", "Total Projected Cost", "Projected Total Cost"
       Formula: Cost to Date + Cost to Complete = Estimated Total Costs
       !! DO NOT use "Total Cost to Date", "Costs to Date", or "Costs Incurred to Date" — those are cost_to_date.
       !! DO NOT use any column that contains a running actual spend figure — estimated_total_costs is always a BUDGET, not an actual.

    3. estimated_gross_profit — Projected profit margin on the job.
       Headers: "Est GP", "Estimated GP", "Gross Profit", "Est Gross Profit", "Projected GP"
       Formula: Contract Price - Estimated Total Costs = Estimated Gross Profit

    4. revenues_earned — Revenue recognized based on percent complete.
       Headers: "Earned Revenue", "Revenue Earned", "Revenues Earned", "Earned Rev",
                "Income Earned", "Billings Earned"
       Formula: Contract Price × (Cost to Date ÷ Estimated Total Costs)

    5. cost_to_date — Actual costs incurred on the job so far.
       Headers: "Cost to Date", "Costs to Date", "Costs Incurred", "Actual Cost", "Cost Incurred to Date",
                "Total Cost to Date"
       This is a cumulative running total. For active jobs it is typically LESS than Estimated Total Costs.
       SPLIT COST COLUMNS: Some documents break costs into sub-periods, e.g.:
         "Prior Period Costs" | "Current Year Costs" | "Total Cost to Date"
       In this case, ignore the sub-period columns entirely. Map ONLY the "Total Cost to Date" column to cost_to_date.
       The sub-period columns are breakdowns of cost_to_date, not separate fields.

    6. gross_profit_to_date — Actual gross profit earned so far.
       Headers: "GP to Date", "Gross Profit to Date", "GP Earned", "Profit to Date"
       Formula: Revenues Earned - Cost to Date

    7. billed_to_date — Total amount invoiced to the owner.
       Headers: "Billed to Date", "Billings to Date", "Total Billed", "Progress Billings", "Amount Billed"

    8. cost_to_complete — Remaining costs needed to finish the job.
       Headers: "Cost to Complete", "CTC", "Est Cost to Complete", "Remaining Cost", "Costs to Complete"
       Formula: Estimated Total Costs - Cost to Date
       For completed jobs this should be 0 or near 0.

    9. under_billings — Amount earned but not yet billed (asset).
       Only populated when Revenues Earned > Billed to Date.
       Headers: "Under Billings", "Underbillings", "CIE", "Costs in Excess", "Unbilled Revenue"

    10. over_billings — Amount billed in excess of earnings (liability).
        Only populated when Billed to Date > Revenues Earned.
        Headers: "Over Billings", "Overbillings", "BIE", "Billings in Excess", "Deferred Revenue"

    MULTI-PAGE DOCUMENTS:
    If the schedule spans multiple pages, continue extracting all job rows. Ignore subtotal rows mid-document (page subtotals, division subtotals). Only treat the final "TOTALS" or "GRAND TOTAL" row at the very end as the totals object.

    PRE-OUTPUT VERIFICATION:
    Before returning, check every row satisfies:
      (a) Cost to Date + Cost to Complete ≈ Estimated Total Costs
      (b) Contract Price - Estimated Total Costs ≈ Estimated Gross Profit
    If a row fails both checks, re-examine the source document — you likely mapped a column incorrectly.

    RULES:
    - Values in parentheses like (100) are negative: -100
    - Missing or blank fields are 0
    - All values are plain numbers, no currency symbols or commas
    - Job IDs may be numeric or alphanumeric — extract exactly as shown

    Return this exact JSON structure:
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
    """

    SYSTEM_PROMPT = (
        "You are a precise financial data extraction engine. Your sole function is to extract "
        "structured data from construction Work-In-Progress (WIP) schedules and return it as valid JSON. "
        "You have deep knowledge of construction accounting: job costing, percent complete billing, "
        "earned value, and the algebraic relationships between WIP columns. You never infer, estimate, "
        "or fabricate values — you extract only what is explicitly present in the document. "
        "Return ONLY the raw JSON object with no markdown, no commentary, no preamble."
    )

    # Inject feedback from a prior failed run if present
    active_prompt = prompt
    if state.extraction_feedback:
        active_prompt = (
            f"FEEDBACK FROM PRIOR EXTRACTION ATTEMPT:\n"
            f"{state.extraction_feedback}\n\n"
            "Apply this feedback carefully before extracting. Pay special attention to the "
            "flagged column(s) and verify your mapping is correct before outputting.\n\n"
        ) + prompt

    def _run_extraction(model_name: str) -> str:
        """Run a single extraction attempt with the given model. Returns raw text."""
        response = client.generate_content(
            prompt=active_prompt,
            model_name=model_name,
            pdf_bytes=file_bytes,
            response_mime_type="application/json",
            tracker=tracker,
            system_prompt=SYSTEM_PROMPT,
        )
        return response.text or ""

    raw_text = ""
    actual_model = state.model_name
    try:
        if state.model_name == OPTIMAL_MIX_KEY:
            # Tier 1: Flash-Lite
            print(f"--- OPTIMAL MIX: trying {OPTIMAL_MIX_PRIMARY} ---")
            raw_text = _run_extraction(OPTIMAL_MIX_PRIMARY)
            actual_model = OPTIMAL_MIX_PRIMARY
            _diag("llm_call_tier1", "OK", f"{OPTIMAL_MIX_PRIMARY}: {len(raw_text)} chars")

            # Quick validation: can we even parse it?
            try:
                parse_json_safely(raw_text)
            except Exception:
                # Unparseable — escalate immediately
                print(f"--- OPTIMAL MIX: tier 1 unparseable, escalating to {OPTIMAL_MIX_FALLBACK} ---")
                raw_text = _run_extraction(OPTIMAL_MIX_FALLBACK)
                actual_model = OPTIMAL_MIX_FALLBACK
                _diag("llm_call_tier2", "OK", f"escalated to {OPTIMAL_MIX_FALLBACK}: {len(raw_text)} chars")
        else:
            raw_text = _run_extraction(state.model_name)
            _diag("llm_call", "OK", f"{len(raw_text)} chars returned")

        diagnostics["actual_model"] = actual_model

    except Exception as e:
        _diag("llm_call", "FAIL", str(e))
        diagnostics["raw_response_preview"] = ""
        return {"processed_data": [], "totals_row": None, "metrics_data": tracker.get_metrics(), "extraction_diagnostics": diagnostics}

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

    data = None
    _row_field_keys = {"job_id", "total_contract_price", "cost_to_date"}

    if isinstance(parsed, dict):
        data = parsed
    elif isinstance(parsed, list) and len(parsed) > 0:
        if isinstance(parsed[0], dict):
            data = {"rows": parsed}
            _diag("json_normalize", "OK", f"Wrapped top-level array ({len(parsed)} dicts) into rows")
        else:
            _diag("json_parse", "FAIL", f"Top-level list but items are {type(parsed[0]).__name__}, not dicts")

    if data is None:
        return {"processed_data": [], "totals_row": None, "metrics_data": tracker.get_metrics(), "extraction_diagnostics": diagnostics}

    raw_rows = []
    totals_data = data.get("totals")

    candidate = data.get("rows")
    if isinstance(candidate, list) and len(candidate) > 0:
        raw_rows = candidate
        _diag("row_locate", "OK", f"Found {len(raw_rows)} rows under 'rows' key")

    if not raw_rows:
        for alt_key in ("data", "jobs", "wip_rows", "schedule", "wip_schedule", "job_rows", "wip_data", "extracted_data"):
            candidate = data.get(alt_key)
            if isinstance(candidate, list) and len(candidate) > 0:
                raw_rows = candidate
                _diag("row_locate", "OK", f"Found {len(raw_rows)} rows under '{alt_key}' key")
                break

    if not raw_rows:
        for key, val in data.items():
            if key == "totals":
                continue
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                if _row_field_keys.issubset(set(val[0].keys())):
                    raw_rows = val
                    _diag("row_locate", "OK", f"Found {len(raw_rows)} row-like dicts under '{key}' key")
                    break

    if not raw_rows and _row_field_keys.issubset(set(data.keys())):
        first_val = data.get("job_id")
        non_meta_data = {k: v for k, v in data.items() if k != "totals"}

        if isinstance(first_val, list):
            num_rows = len(first_val)
            raw_rows = [
                {k: (v[i] if isinstance(v, list) and i < len(v) else v) for k, v in non_meta_data.items()}
                for i in range(num_rows)
            ]
            _diag("row_locate", "OK", f"Converted columnar format ({num_rows} rows)")
        else:
            raw_rows = [non_meta_data]
            _diag("row_locate", "OK", "Wrapped single flat object as 1 row")

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

            if failed_stage == "file_read":
                struct_msg = f"File Read Failed: {failure_chain[0] if failure_chain else 'unknown error'}"
            elif failed_stage == "llm_call":
                struct_msg = f"Model Call Failed: {failure_chain[0] if failure_chain else 'API error'}"
            elif failed_stage == "json_parse":
                preview = raw_preview[:200].replace('\n', ' ').strip()
                struct_msg = f"JSON Parse Failed — model returned: \"{preview}...\""
            elif failed_stage in ("row_locate", "row_parse"):
                last_detail = failure_chain[-1] if failure_chain else "unknown structure"
                struct_msg = f"Data structure issue: {last_detail}"
            else:
                struct_msg = f"Extraction failed at: {failed_stage}"

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
        all_root_cause_results: List[Dict[str, Any]] = []

        for r in rows:
            row_errors = run_validations(r, validations)
            pre_correction_errors.extend(row_errors)
            if row_errors:
                root_cause_result = resolve_root_causes(r, row_errors, validations)
                all_root_cause_results.append({"job_id": r.job_id, **root_cause_result})
                for err in row_errors:
                    vname = err["validation"]
                    rc_info = root_cause_result["by_validation"].get(vname, {})
                    err["is_cascade"] = rc_info.get("is_cascade", False)
                    err["root_cause_field"] = rc_info.get("root_cause_field")
                    err["root_cause_value"] = rc_info.get("root_cause_value")
                    err["root_resolutions"] = rc_info.get("resolutions", 0)
                suggestions = suggest_corrections(r, row_errors)
                all_correction_suggestions.extend(suggestions)

        # Portfolio-level column error detection — must run before column swaps
        # so corrections are applied before the swap check re-reads row values
        portfolio_column_errors = detect_portfolio_column_errors(
            rows, all_root_cause_results, len(pre_correction_errors), validations
        )

        column_swaps = detect_column_swaps(rows)
        correction_log = apply_corrections(rows, column_swaps, all_correction_suggestions)

        for r in rows:
            variance = r.revenues_earned - r.billed_to_date
            if variance > 0:
                r.under_billings_calc = variance
                r.over_billings_calc = 0.0
            else:
                r.under_billings_calc = 0.0
                r.over_billings_calc = abs(variance)

        post_correction_errors: List[Dict[str, Any]] = []
        for r in rows:
            row_errors = run_validations(r, validations)
            post_correction_errors.extend(row_errors)

        all_validation_errors = post_correction_errors

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

        _totals_fields = [
            ("total_contract_price", "Contract Value"),
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
        totals_detail_rows: List[Dict[str, Any]] = []

        if extracted_totals:
            mismatches = []
            for field_name, display_name in _totals_fields:
                calc_val = getattr(calc, field_name, 0.0)
                ext_val = getattr(extracted_totals, field_name, 0.0)
                diff = abs(calc_val - ext_val)
                match = diff < 1000.0

                detail_row = {
                    "label": display_name,
                    "extracted": f"${ext_val:,.0f}",
                    "calculated": f"${calc_val:,.0f}",
                    "match": match,
                }
                if not match:
                    signed_diff = calc_val - ext_val
                    detail_row["delta"] = f"{'+'if signed_diff > 0 else '−'}${abs(signed_diff):,.0f}"
                    mismatches.append((field_name, display_name, calc_val, ext_val, diff))

                totals_detail_rows.append(detail_row)

            if not mismatches:
                totals_pass = True
                totals_msg = f"All {len(_totals_fields)} columns match"
            else:
                passed_count = len(_totals_fields) - len(mismatches)
                totals_msg = f"{len(mismatches)} column{'s' if len(mismatches) != 1 else ''} off"
                for fname, dname, cv, ev, diff in mismatches:
                    totals_details.append({
                        "id": "TOTALS",
                        "field": fname,
                        "msg": f"{dname}: Calc ${cv:,.0f} vs Report ${ev:,.0f} (diff ${diff:,.0f})",
                    })

        surety_risk_context = build_surety_risk_context(rows, calc)
        surety_risk_context["risk_tier"] = compute_portfolio_risk_tier(surety_risk_context)
        risk_rows = build_risk_rows(rows, calc)

        t_uegp = calc.estimated_gross_profit - calc.gross_profit_to_date
        gp_percent = (calc.gross_profit_to_date / calc.revenues_earned * 100) if calc.revenues_earned else 0
        net_ub_ob = calc.under_billings - calc.over_billings
        net_ub_ob_label = f"Under ${net_ub_ob/1000:.0f}k" if net_ub_ob >= 0 else f"Over ${abs(net_ub_ob)/1000:.0f}k"

        struct_pass = len(rows) > 0 and all(r.job_id for r in rows)
        struct_msg = "Structure Valid" if struct_pass else "Missing IDs/Data"

        missing_ids = sum(1 for r in rows if not r.job_id)
        struct_detail_rows = [
            {"label": "Column headers detected", "value": f"10 of 10" if struct_pass else "Incomplete", "match": struct_pass},
            {"label": "Row count", "value": f"{len(rows)} data rows", "match": len(rows) > 0},
            {"label": "Total row detected", "value": "Present" if extracted_totals else "Missing", "match": extracted_totals is not None},
            {"label": "Missing job IDs", "value": f"{missing_ids} found" if missing_ids else "0 found", "match": missing_ids == 0},
        ]

        validation_type_labels = {
            "contract_cost_gp": "Contract − Est Cost = Est GP",
            "cost_to_complete_check": "Est Cost = CTD + CTC",
            "earned_revenue_from_gp": "CTD + GP to Date = Earned Rev",
            "underbilling_overbilling": "UB/OB matches billing variance",
            "earned_revenue_from_poc": "Earned Rev ≈ Contract × POC",
            "remaining_gp_check": "GP to Date ≤ Est GP",
            "gp_percentage_bounds": "GP% within normal range",
        }

        formula_detail_rows = []
        for v in validations:
            pass_count = 0
            fail_count = 0
            for r in rows:
                row_dict = r.model_dump()
                required_present = all(row_dict.get(field) is not None for field in v.requires)
                has_nonzero = any(row_dict.get(field, 0) != 0 for field in v.requires)
                if not required_present or not has_nonzero:
                    continue
                error_msg = v.check(r)
                if error_msg:
                    fail_count += 1
                else:
                    pass_count += 1

            total_checked = pass_count + fail_count
            if total_checked == 0:
                continue

            label = validation_type_labels.get(v.name, v.name)
            if fail_count == 0:
                value_str = f"{pass_count}/{total_checked} rows pass"
            elif fail_count <= 2:
                value_str = f"{pass_count}/{total_checked} rows pass ({fail_count} {'rounding ≤$1' if fail_count == 1 and v.tolerance_value <= 100 else 'issue' + ('s' if fail_count > 1 else '')})"
            else:
                value_str = f"{fail_count}/{total_checked} rows failed"

            formula_detail_rows.append({
                "label": label,
                "value": value_str,
                "match": fail_count == 0,
            })

        errors_by_job: Dict[str, List[Dict[str, Any]]] = {}
        for e in all_validation_errors:
            errors_by_job.setdefault(e["job_id"], []).append(e)

        formula_pass = len(all_validation_errors) == 0
        if portfolio_column_errors:
            ce = portfolio_column_errors[0]
            field_label = ce["field"].replace("_", " ").title()
            formula_msg = (
                f"Column Error: {field_label} — corrected across {ce['rows_corrected']} rows "
                f"({ce['rows_resolved']}/{ce['total_rows']} rows resolved)"
            )
            # Build feedback string for retry — surfaced in the API response so the
            # frontend can pass it back on a retry request
            _pct = int(ce["row_resolution_rate"] * 100)
            _feedback_for_retry = (
                f"{ce['rows_resolved']} of {ce['total_rows']} jobs failed formula validation "
                f"({_pct}% failure rate). This is a strong indicator of a column mapping error "
                f"on the '{field_label}' column. On your next attempt, pay very careful attention "
                f"to the '{field_label}' column. Double-check that you are reading the correct "
                f"column and not an adjacent one with a similar name or value."
            )
        else:
            _feedback_for_retry = ""

        if formula_pass and correction_log:
            formula_msg = f"Column Math Validated (after {len(correction_log)} auto-corrections)"
        elif formula_pass:
            formula_msg = "Column Math Validated"
        else:
            formula_msg = f"Column Math Issues ({len(errors_by_job)} rows)"

        jobs_with_issues: Dict[str, Dict[str, Any]] = {}
        for e in all_validation_errors:
            job_id = e["job_id"]
            jobs_with_issues.setdefault(job_id, {"errors": [], "corrections": []})
            jobs_with_issues[job_id]["errors"].append(
                {
                    "validation": e["validation"],
                    "message": e["message"],
                    "category": e["category"],
                    "root_cause_field": e.get("root_cause_field"),
                    "is_cascade": e.get("is_cascade", False),
                }
            )

        for s in all_correction_suggestions:
            job_id = s["job_id"]
            jobs_with_issues.setdefault(job_id, {"errors": [], "corrections": []})
            applied = any(
                cl["job_id"] == job_id and cl.get("field") == s.get("field")
                for cl in correction_log if cl["type"] == "digit_fix"
            )
            suggested_display = f"${s['suggested_value']:,.0f}" if s.get('suggested_value') is not None else "Manual review needed"
            jobs_with_issues[job_id]["corrections"].append(
                {
                    "field": s["field"],
                    "current": f"${s['current_value']:,.0f}",
                    "suggested": suggested_display,
                    "confidence": s["confidence"],
                    "cross_validation_count": s.get("cross_validation_count", 1),
                    "applied": applied,
                    "reasoning": s["reasoning"],
                }
            )

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

        def _build_chart_data(rows: List[CalculatedWipRow], calc: WipTotals) -> Dict[str, Any]:
            billing_rows = []
            for r in rows:
                net = r.over_billings_calc - r.under_billings_calc
                if abs(net) > 1000:
                    short_name = (r.job_name or r.job_id)[:18]
                    billing_rows.append({"name": short_name, "value": round(net / 1000)})
            billing_rows.sort(key=lambda x: x["value"])
            billing_chart = billing_rows 

            risk_tier_lookup = {rr["jobId"]: rr["riskTier"] for rr in risk_rows}
            exposure_rows = []
            for r in rows:
                remaining = r.cost_to_complete
                if remaining > 10000:
                    tier = risk_tier_lookup.get(r.job_id, "NONE")
                    short_name = (r.job_name or r.job_id)[:18]
                    exposure_rows.append({
                        "name": short_name,
                        "completed": round(r.cost_to_date / 1000),
                        "remaining": round(remaining / 1000),
                        "tier": tier if tier != "NONE" else "LOW",
                    })
            exposure_rows.sort(key=lambda x: x["remaining"], reverse=True)
            exposure_chart = exposure_rows

            margin_rows = []
            for r in rows:
                if r.total_contract_price > 100000 and r.estimated_total_costs > 0:
                    est_gp_pct = round((r.estimated_gross_profit / r.total_contract_price) * 100, 1)
                    poc = r.cost_to_date / r.estimated_total_costs
                    if poc > 0.05:
                        actual_gp_pct = round((r.gross_profit_to_date / r.revenues_earned) * 100, 1) if r.revenues_earned else 0
                        short_name = (r.job_name or r.job_id)[:18]
                        margin_rows.append({
                            "name": short_name,
                            "estimated": est_gp_pct,
                            "actual": actual_gp_pct,
                        })
            margin_rows.sort(key=lambda x: x["estimated"] - x["actual"], reverse=True)
            margin_chart = margin_rows

            return {
                "billing": billing_chart,
                "exposure": exposure_chart,
                "margin": margin_chart,
            }

        charts = _build_chart_data(rows, calc)

        kpis = [
            {"label": "Contract Value", "value": f"${calc.total_contract_price/1e6:.1f}M"},
            {"label": "UEGP", "value": f"${t_uegp/1e6:.1f}M"},
            {"label": "CTC", "value": f"${calc.cost_to_complete/1e6:.1f}M"},
            {"label": "Earned Rev", "value": f"${calc.revenues_earned/1e6:.1f}M"},
            {"label": "GP %", "value": f"{gp_percent:.1f}%"},
            {"label": "Net Position", "value": net_ub_ob_label, "negative": net_ub_ob >= 0},
        ]

        widget_data = {
            "validations": {
                "structural": {
                    "passed": struct_pass,
                    "message": struct_msg,
                    "details": struct_detail_rows,
                },
                "formulaic": {
                    "passed": formula_pass,
                    "message": formula_msg,
                    "details": formula_detail_rows,
                    "jobIssues": [] if portfolio_column_errors else corrections_display,
                    "columnErrors": portfolio_column_errors,
                    "retryFeedback": _feedback_for_retry,
                },
                "totals": {
                    "passed": totals_pass,
                    "message": totals_msg,
                    "details": totals_detail_rows if totals_detail_rows else totals_details,
                },
            },
            "kpis": kpis,
            "charts": charts,
            "corrections_applied": correction_log,
            "riskTier": surety_risk_context["risk_tier"],
            "riskRowsAll": risk_rows,
            "metrics": {
                "row1_1": {"label": "Contract Value", "value": f"${calc.total_contract_price/1000000:.2f}M"},
                "row1_2": {"label": "UEGP", "value": f"${t_uegp/1000000:.2f}M"},
                "row1_3": {"label": "CTC", "value": f"${calc.cost_to_complete/1000000:.2f}M"},
                "row2_1": {"label": "Earned Rev", "value": f"${calc.revenues_earned/1000000:.2f}M"},
                "row2_2": {"label": "GP %", "value": f"{gp_percent:.1f}%"},
                "row2_3": {"label": "Net UB / OB", "value": net_ub_ob_label},
            },
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
# 8. NARRATIVE NODE
# ==========================================

SURETY_NARRATIVE_PROMPT = """You are a surety underwriting analyst. Write a 2-4 sentence executive abstract of this WIP schedule for the chief underwriter.

PERSPECTIVE: We bonded this contractor. Evaluate everything through: "What does this mean for our recovery if they default?"

PRIORITY: (1) Underbilling exposure = work done, cash not collected = direct recovery risk. (2) Loss jobs / GP fade = eroding net worth. (3) Concentration risk. (4) Overbilling is less concerning (cash collected ahead of work).

DATA QUALITY: {correction_context}

RULES:
- Exactly 2-4 sentences. No more.
- Lead with the single most important insight.
- Connect dots between signals — don't just restate numbers the dashboard already shows.
- End with what to watch or ask for at next review.
- No bullets, no headers, no bold, no preamble.

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
            max_tokens=500,
            tracker=tracker,
            system_prompt="You are a surety underwriting analyst. Write exactly 2-4 sentences of prose. No bullets, no headers. Be specific and insightful.",
        )

        narrative = response.text.strip()

        if response.finish_reason:
            print(f"    Narrative finish_reason: {response.finish_reason}")
            print(f"    Narrative length: {len(narrative)} chars, ~{response.output_tokens} tokens")

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
# 9. OUTPUT NODE
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
            "actual_model": state.extraction_diagnostics.get("actual_model", state.model_name) if state.extraction_diagnostics else state.model_name,
        }

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
