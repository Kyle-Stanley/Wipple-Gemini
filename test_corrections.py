"""
WIP Correction System Test Harness

Injects synthetic OCR errors into clean extracted data and verifies
the detection/correction pipeline catches them.

Usage:
    python test_corrections.py                    # Run all tests with synthetic data
    python test_corrections.py --from-json FILE   # Inject errors into real extracted data
"""

import json
import copy
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Import the correction system
from wip_agent import (
    CalculatedWipRow,
    WipTotals,
    build_validations,
    run_validations,
    detect_column_swaps,
    apply_corrections,
    digit_change_score,
    build_risk_rows,
    build_surety_risk_context,
    compute_portfolio_risk_tier,
)


# ==========================================
# SYNTHETIC TEST DATA (realistic WIP rows)
# ==========================================

CLEAN_ROWS = [
    {
        "job_id": "2024-001",
        "job_name": "Highway Bridge Rehab",
        "total_contract_price": 4500000,
        "estimated_total_costs": 3800000,
        "estimated_gross_profit": 700000,
        "revenues_earned": 2700000,
        "cost_to_date": 2280000,
        "gross_profit_to_date": 420000,
        "billed_to_date": 2500000,
        "cost_to_complete": 1520000,
        "under_billings": 200000,
        "over_billings": 0,
    },
    {
        "job_id": "2024-002",
        "job_name": "School Renovation Phase 2",
        "total_contract_price": 1200000,
        "estimated_total_costs": 1050000,
        "estimated_gross_profit": 150000,
        "revenues_earned": 900000,
        "cost_to_date": 787500,
        "gross_profit_to_date": 112500,
        "billed_to_date": 850000,
        "cost_to_complete": 262500,
        "under_billings": 50000,
        "over_billings": 0,
    },
    {
        "job_id": "2024-003",
        "job_name": "Municipal Water Treatment",
        "total_contract_price": 8200000,
        "estimated_total_costs": 7100000,
        "estimated_gross_profit": 1100000,
        "revenues_earned": 1640000,
        "cost_to_date": 1420000,
        "gross_profit_to_date": 220000,
        "billed_to_date": 1800000,
        "cost_to_complete": 5680000,
        "under_billings": 0,
        "over_billings": 160000,
    },
    {
        "job_id": "2024-004",
        "job_name": "Commercial HVAC Install",
        "total_contract_price": 650000,
        "estimated_total_costs": 580000,
        "estimated_gross_profit": 70000,
        "revenues_earned": 585000,
        "cost_to_date": 522000,
        "gross_profit_to_date": 63000,
        "billed_to_date": 600000,
        "cost_to_complete": 58000,
        "under_billings": 0,
        "over_billings": 15000,
    },
    {
        "job_id": "2024-005",
        "job_name": "Parking Garage Expansion",
        "total_contract_price": 3100000,
        "estimated_total_costs": 2750000,
        "estimated_gross_profit": 350000,
        "revenues_earned": 1550000,
        "cost_to_date": 1375000,
        "gross_profit_to_date": 175000,
        "billed_to_date": 1400000,
        "cost_to_complete": 1375000,
        "under_billings": 150000,
        "over_billings": 0,
    },
]


# ==========================================
# ERROR INJECTION
# ==========================================

@dataclass
class InjectedError:
    """Record of a synthetic error for verification."""
    job_id: str
    field: str
    original_value: float
    corrupted_value: float
    error_type: str
    description: str


def inject_dropped_digit(rows: List[Dict], job_idx: int, field: str) -> InjectedError:
    """Drop a trailing digit: 1520000 → 152000"""
    original = rows[job_idx][field]
    s = str(int(abs(original)))
    corrupted = float(s[:-1]) * (-1 if original < 0 else 1)
    rows[job_idx][field] = corrupted
    return InjectedError(
        job_id=rows[job_idx]["job_id"], field=field,
        original_value=original, corrupted_value=corrupted,
        error_type="dropped_trailing_digit",
        description=f"{field}: {original:,.0f} → {corrupted:,.0f}",
    )


def inject_dropped_leading_digit(rows: List[Dict], job_idx: int, field: str) -> InjectedError:
    """Drop leading digit: 150000 → 50000"""
    original = rows[job_idx][field]
    s = str(int(abs(original)))
    corrupted = float(s[1:]) * (-1 if original < 0 else 1)
    rows[job_idx][field] = corrupted
    return InjectedError(
        job_id=rows[job_idx]["job_id"], field=field,
        original_value=original, corrupted_value=corrupted,
        error_type="dropped_leading_digit",
        description=f"{field}: {original:,.0f} → {corrupted:,.0f}",
    )


def inject_single_digit_misread(rows: List[Dict], job_idx: int, field: str, pos: int = 0, to_digit: str = "3") -> InjectedError:
    """Change one digit: 2280000 → 3280000"""
    original = rows[job_idx][field]
    s = list(str(int(abs(original))))
    old_digit = s[pos]
    s[pos] = to_digit
    corrupted = float("".join(s)) * (-1 if original < 0 else 1)
    rows[job_idx][field] = corrupted
    return InjectedError(
        job_id=rows[job_idx]["job_id"], field=field,
        original_value=original, corrupted_value=corrupted,
        error_type="single_digit_misread",
        description=f"{field}: {original:,.0f} → {corrupted:,.0f} (digit {pos}: {old_digit}→{to_digit})",
    )


def inject_column_swap(rows: List[Dict], job_idx: int) -> InjectedError:
    """Swap cost_to_date and cost_to_complete"""
    ctd = rows[job_idx]["cost_to_date"]
    ctc = rows[job_idx]["cost_to_complete"]
    rows[job_idx]["cost_to_date"] = ctc
    rows[job_idx]["cost_to_complete"] = ctd
    return InjectedError(
        job_id=rows[job_idx]["job_id"], field="cost_to_date↔cost_to_complete",
        original_value=ctd, corrupted_value=ctc,
        error_type="column_swap",
        description=f"CTD/CTC swapped: CTD {ctd:,.0f}↔{ctc:,.0f} CTC",
    )


def inject_dropped_zero(rows: List[Dict], job_idx: int, field: str) -> InjectedError:
    """Drop a trailing zero: 1100000 → 110000"""
    original = rows[job_idx][field]
    s = str(int(abs(original)))
    if s.endswith("0"):
        corrupted = float(s[:-1]) * (-1 if original < 0 else 1)
    else:
        corrupted = original / 10
    rows[job_idx][field] = corrupted
    return InjectedError(
        job_id=rows[job_idx]["job_id"], field=field,
        original_value=original, corrupted_value=corrupted,
        error_type="dropped_zero",
        description=f"{field}: {original:,.0f} → {corrupted:,.0f}",
    )


# ==========================================
# TEST SCENARIOS
# ==========================================

def build_test_scenarios() -> List[Tuple[str, List[Dict], List[InjectedError]]]:
    """Build a suite of error injection scenarios."""
    scenarios = []

    # Scenario 1: Single dropped trailing digit on cost_to_complete
    rows = copy.deepcopy(CLEAN_ROWS)
    errors = [inject_dropped_digit(rows, 0, "cost_to_complete")]
    scenarios.append(("Dropped trailing digit (CTC)", rows, errors))

    # Scenario 2: Column swap on CTD/CTC
    rows = copy.deepcopy(CLEAN_ROWS)
    errors = [inject_column_swap(rows, 0)]
    scenarios.append(("Column swap CTD↔CTC", rows, errors))

    # Scenario 3: Single digit misread on estimated_total_costs
    rows = copy.deepcopy(CLEAN_ROWS)
    errors = [inject_single_digit_misread(rows, 1, "estimated_total_costs", pos=0, to_digit="2")]
    scenarios.append(("Single digit misread (ETC)", rows, errors))

    # Scenario 4: Dropped leading digit on revenues_earned
    rows = copy.deepcopy(CLEAN_ROWS)
    errors = [inject_dropped_leading_digit(rows, 2, "revenues_earned")]
    scenarios.append(("Dropped leading digit (Rev Earned)", rows, errors))

    # Scenario 5: Multiple errors on same row
    rows = copy.deepcopy(CLEAN_ROWS)
    errors = [
        inject_dropped_zero(rows, 3, "estimated_gross_profit"),
        inject_single_digit_misread(rows, 3, "cost_to_date", pos=0, to_digit="6"),
    ]
    scenarios.append(("Multiple errors same row", rows, errors))

    # Scenario 6: Errors across different rows
    rows = copy.deepcopy(CLEAN_ROWS)
    errors = [
        inject_dropped_digit(rows, 0, "cost_to_date"),
        inject_column_swap(rows, 2),
        inject_single_digit_misread(rows, 4, "estimated_gross_profit", pos=0, to_digit="4"),
    ]
    scenarios.append(("Multi-row mixed errors", rows, errors))

    # Scenario 7: Clean data (no errors — should produce zero corrections)
    rows = copy.deepcopy(CLEAN_ROWS)
    scenarios.append(("Clean data (no errors)", rows, []))

    return scenarios


# ==========================================
# RUNNER
# ==========================================

def run_scenario(name: str, raw_rows: List[Dict], injected: List[InjectedError]) -> Dict[str, Any]:
    """Run the correction pipeline on corrupted data and report results."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")

    # Parse rows
    rows = [CalculatedWipRow(**r) for r in raw_rows]

    # Compute UB/OB
    for r in rows:
        variance = r.revenues_earned - r.billed_to_date
        r.under_billings_calc = max(0, variance)
        r.over_billings_calc = max(0, -variance)

    # Print injected errors
    if injected:
        print(f"\n  INJECTED {len(injected)} error(s):")
        for e in injected:
            print(f"    [{e.error_type}] {e.description}")
    else:
        print(f"\n  NO ERRORS INJECTED (control test)")

    # Run validations (pre-correction)
    validations = build_validations()
    pre_errors = []
    for r in rows:
        pre_errors.extend(run_validations(r, validations))

    print(f"\n  PRE-CORRECTION:")
    print(f"    Validation errors: {len(pre_errors)}")
    for e in pre_errors[:5]:
        print(f"      [{e['job_id']}] {e['validation']}: {e['message']}")

    # Detect column swaps
    swaps = detect_column_swaps(rows)
    print(f"    Column swaps detected: {len(swaps)}")
    for s in swaps:
        print(f"      [{s['job_id']}] {s.get('detail', '')}")

    # Apply corrections
    correction_log = apply_corrections(rows, swaps, [])
    print(f"\n  CORRECTIONS APPLIED: {len(correction_log)}")
    for c in correction_log:
        print(f"    [{c['job_id']}] {c['type']}: {c.get('detail', c.get('field', ''))}")

    # Re-validate
    post_errors = []
    for r in rows:
        post_errors.extend(run_validations(r, validations))

    print(f"\n  POST-CORRECTION:")
    print(f"    Remaining errors: {len(post_errors)}")
    for e in post_errors[:5]:
        print(f"      [{e['job_id']}] {e['validation']}: {e['message']}")

    # Score results
    detected = 0
    corrected = 0
    missed = 0

    for inj in injected:
        was_detected = any(
            e["job_id"] == inj.job_id for e in pre_errors
        ) or any(
            s["job_id"] == inj.job_id for s in swaps
        )
        was_corrected = any(
            c["job_id"] == inj.job_id for c in correction_log
        )

        if was_corrected:
            corrected += 1
            detected += 1
        elif was_detected:
            detected += 1
        else:
            missed += 1

    result = {
        "scenario": name,
        "injected": len(injected),
        "detected": detected,
        "corrected": corrected,
        "missed": missed,
        "pre_validation_errors": len(pre_errors),
        "post_validation_errors": len(post_errors),
        "false_positives": len(post_errors) if not injected else 0,
    }

    status = "✓ PASS" if missed == 0 and (not injected or detected > 0) else "✗ FAIL"
    print(f"\n  RESULT: {status}")
    print(f"    Injected: {len(injected)} | Detected: {detected} | Corrected: {corrected} | Missed: {missed}")

    return result


def run_digit_score_tests():
    """Verify digit_change_score handles all OCR error patterns."""
    print(f"\n{'='*70}")
    print("DIGIT CHANGE SCORE UNIT TESTS")
    print(f"{'='*70}")

    tests = [
        (1520000, 152000, 1, "dropped trailing digit"),
        (2280000, 228000, 1, "dropped trailing zero"),
        (1640000, 640000, 1, "dropped leading digit"),
        (150000, 15000, 1, "dropped trailing zero (order of magnitude)"),
        (11000, 1100, 1, "dropped trailing zero"),
        (2280000, 3280000, 1, "single digit misread"),
        (1050000, 2050000, 1, "single digit misread"),
        (70000, 7000, 1, "dropped trailing zero"),
        (350000, 450000, 1, "single digit misread"),
        (150000, 1500, 50, "two orders magnitude — unlikely OCR"),
        (-15000, 15000, 50, "sign flip"),
        (1234567, 1234567, 0, "identical values"),
    ]

    passed = 0
    for current, target, expected, desc in tests:
        actual = digit_change_score(current, target)
        ok = actual == expected
        passed += ok
        status = "✓" if ok else "✗"
        print(f"  {status} {desc}: score({current:>10,} → {target:>10,}) = {actual} (expected {expected})")

    print(f"\n  {passed}/{len(tests)} tests passed")


def run_risk_type_tests():
    """Verify all risk types fire correctly."""
    print(f"\n{'='*70}")
    print("RISK TYPE DETECTION TESTS")
    print(f"{'='*70}")

    # Test data designed to trigger each risk type
    test_rows_data = [
        {   # Loss Job
            "job_id": "RISK-001", "job_name": "Losing Money",
            "total_contract_price": 500000, "estimated_total_costs": 550000,
            "estimated_gross_profit": -50000, "revenues_earned": 300000,
            "cost_to_date": 330000, "gross_profit_to_date": -30000,
            "billed_to_date": 280000, "cost_to_complete": 220000,
            "under_billings": 20000, "over_billings": 0,
        },
        {   # Severe Underbilling
            "job_id": "RISK-002", "job_name": "Underbilled Project",
            "total_contract_price": 2000000, "estimated_total_costs": 1700000,
            "estimated_gross_profit": 300000, "revenues_earned": 1200000,
            "cost_to_date": 1020000, "gross_profit_to_date": 180000,
            "billed_to_date": 900000, "cost_to_complete": 680000,
            "under_billings": 300000, "over_billings": 0,
        },
        {   # Early Stage + Concentration
            "job_id": "RISK-003", "job_name": "New Big Job",
            "total_contract_price": 10000000, "estimated_total_costs": 8500000,
            "estimated_gross_profit": 1500000, "revenues_earned": 800000,
            "cost_to_date": 680000, "gross_profit_to_date": 120000,
            "billed_to_date": 900000, "cost_to_complete": 7820000,
            "under_billings": 0, "over_billings": 100000,
        },
        {   # Cost Overrun Signal + Thin Margin
            "job_id": "RISK-004", "job_name": "Squeezed Margins",
            "total_contract_price": 800000, "estimated_total_costs": 790000,
            "estimated_gross_profit": 10000, "revenues_earned": 600000,
            "cost_to_date": 592500, "gross_profit_to_date": 7500,
            "billed_to_date": 580000, "cost_to_complete": 197500,
            "under_billings": 20000, "over_billings": 0,
        },
        {   # Billing Lag
            "job_id": "RISK-005", "job_name": "Slow Biller",
            "total_contract_price": 1500000, "estimated_total_costs": 1300000,
            "estimated_gross_profit": 200000, "revenues_earned": 900000,
            "cost_to_date": 780000, "gross_profit_to_date": 120000,
            "billed_to_date": 550000, "cost_to_complete": 520000,
            "under_billings": 350000, "over_billings": 0,
        },
    ]

    rows = [CalculatedWipRow(**r) for r in test_rows_data]
    for r in rows:
        variance = r.revenues_earned - r.billed_to_date
        r.under_billings_calc = max(0, variance)
        r.over_billings_calc = max(0, -variance)

    calc = WipTotals()
    for r in rows:
        calc.total_contract_price += r.total_contract_price
        calc.estimated_total_costs += r.estimated_total_costs
        calc.cost_to_date += r.cost_to_date

    risk_rows = build_risk_rows(rows, calc)

    expected_tags = {
        "RISK-001": ["Loss Job"],
        "RISK-002": ["Severe Underbilling"],
        "RISK-003": ["Early Stage Exposure", "Concentration Risk"],
        "RISK-004": ["Cost Overrun Signal", "Thin Margin"],
        "RISK-005": ["Billing Lag"],
    }

    passed = 0
    for job_id, expected in expected_tags.items():
        risk_row = next((r for r in risk_rows if r["jobId"] == job_id), None)
        actual_tags = risk_row["riskTags"].split(", ") if risk_row else []

        all_found = all(tag in actual_tags for tag in expected)
        status = "✓" if all_found else "✗"
        passed += all_found

        print(f"  {status} {job_id}: expected {expected}")
        print(f"    actual: {actual_tags}")

    print(f"\n  {passed}/{len(expected_tags)} risk type tests passed")

    # Also test portfolio risk context
    risk_context = build_surety_risk_context(rows, calc)
    tier = compute_portfolio_risk_tier(risk_context)
    print(f"\n  Portfolio risk tier: {tier}")
    print(f"  Early stage jobs: {risk_context['exposure_risk']['early_stage_count']}")
    print(f"  Billing lag jobs: {risk_context['cash_risk']['billing_lag_count']}")
    print(f"  Total billing gap: ${risk_context['cash_risk']['total_billing_gap']:,.0f}")


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 70)
    print("WIP CORRECTION SYSTEM TEST HARNESS")
    print("=" * 70)

    # 1. Digit score unit tests
    run_digit_score_tests()

    # 2. Error injection scenarios
    scenarios = build_test_scenarios()
    results = []
    for name, rows, errors in scenarios:
        result = run_scenario(name, rows, errors)
        results.append(result)

    # 3. Risk type tests
    run_risk_type_tests()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total_injected = sum(r["injected"] for r in results)
    total_detected = sum(r["detected"] for r in results)
    total_corrected = sum(r["corrected"] for r in results)
    total_missed = sum(r["missed"] for r in results)

    for r in results:
        status = "✓" if r["missed"] == 0 else "✗"
        print(f"  {status} {r['scenario']}: {r['detected']}/{r['injected']} detected, {r['corrected']} corrected")

    print(f"\n  TOTAL: {total_detected}/{total_injected} detected, {total_corrected} corrected, {total_missed} missed")

    if total_missed > 0:
        print("\n  ⚠ SOME ERRORS WERE NOT DETECTED — correction system needs improvement")
        sys.exit(1)
    else:
        print("\n  ✓ All injected errors were detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
