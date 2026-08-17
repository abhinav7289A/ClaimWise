"""Deterministic claim-settlement arithmetic. No LLM touches these numbers.

**Why this is Python and not a prompt.** A wrong reimbursement figure is not a
bad completion, it is a liability — and a number a model produced in its head
cannot be audited, reproduced, or unit-tested. CLAUDE.md forbids LLM arithmetic
outright for that reason. The model's job is to read the policy and decide
*which* values apply; turning those values into a rupee figure is this module's,
and it does it the same way every time.

**The order of operations is the domain.** Indian health claims are not "bill
times percentage". Each rule fires on the balance left by the previous one, and
reordering two steps changes the answer by tens of thousands of rupees:

    1. Waiting period      -> ineligible means payable is zero, stop
    2. Non-payable items   -> consumables, gloves, admin (~5-10% of a bill)
    3. Room-rent proportionate deduction
    4. Sub-limit           -> per-procedure cap
    5. Deductible
    6. Co-payment
    7. Sum insured         -> final ceiling

**Step 3 is the one nobody expects, and the reason this tool earns its place.**
Choose a hospital room above your eligible category and most Indian policies
scale down *every associated charge* — surgeon's fees, ICU, investigations — by
the ratio of eligible rent to actual rent. A ₹6,000 room against a ₹4,000
entitlement can remove a third of the whole claim, and no user predicts it.
Applying co-payment before this instead of after produces a different, wrong
number.

**Every policy value carries the page it came from.** A rate arrives as
`PolicyTerm(value=10.0, source_page=39, ...)`, never a bare float. That makes the
arithmetic auditable end to end — the answer can say "10% co-payment applies
[p.39]" — and it extends Phase 1's citation-validity machinery to cover the
numbers as well as the prose.

**Absent is not zero.** A missing co-payment term does not mean "no co-payment",
it means nobody read one. Absent terms are recorded in `assumptions` and the
result is flagged `complete=False`, so an answer built on them can say so. A
silent default of 0% produces a confidently wrong figure, which is the failure
mode this entire phase exists to prevent.

Usage:
    python -m phase3_agents.claims_calculator --help
    python -m phase3_agents.claims_calculator --self-test
    python -m phase3_agents.claims_calculator --demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger("claimwise.claims_calculator")

# Rupees. Money is rounded to whole units at each step because that is what
# settlement letters do, and a breakdown whose lines do not add up to its total
# is worse than useless in front of a user disputing a claim.
ROUNDING = 0


class InvalidClaimInput(ValueError):
    """Raised when the inputs cannot describe a real claim.

    Distinct from a missing policy term, which is recoverable and recorded as an
    assumption. This means the request itself is malformed — a negative bill, a
    co-payment above 100% — and continuing would produce a number that looks
    plausible and means nothing.
    """


@dataclass(frozen=True)
class PolicyTerm:
    """One value read from a policy, with the page it was read from.

    Attributes:
        value: The number itself — a percentage, a rupee cap, or a month count.
        source_page: 1-indexed PDF page. Required, because an uncitable number
            cannot appear in an answer this project is willing to give.
        label: Human description for the breakdown, e.g. "co-payment 10%".
    """

    value: float
    source_page: int
    label: str

    def __post_init__(self) -> None:
        """Validate the term.

        Raises:
            InvalidClaimInput: If the page number is not a positive integer.
        """
        if self.source_page < 1:
            raise InvalidClaimInput(
                f"PolicyTerm {self.label!r} has no usable source page "
                f"({self.source_page}). Every number in an answer must be citable."
            )


@dataclass
class ClaimRequest:
    """Everything needed to settle one hospitalisation claim.

    Bill facts come from the user or an uploaded bill; policy terms come from
    retrieval and each carry a page. Terms left as None are treated as "not
    read" rather than "not applicable" — see the module docstring.

    Attributes:
        claimed_amount: Total hospital bill, in rupees.
        non_payable_amount: Consumables and similar heads the policy never pays.
        room_rent_per_day: Actual room rent charged per day.
        hospitalisation_days: Nights stayed; used only to sanity-check room rent.
        policy_age_months: Months since inception, for the waiting-period test.
        exempt_from_proportion: Charges the policy excludes from the room-rent
            proportionate deduction. Many policies exempt pharmacy and
            consumables; the value varies by wording, so it is an input rather
            than an assumption baked into the rule.
        sum_insured: Overall ceiling.
        co_pay_percent: Percentage the insured bears.
        room_rent_cap_per_day: Eligible room rent per day.
        sub_limit: Procedure-specific cap.
        deductible: Fixed amount borne by the insured before the policy pays.
        waiting_period_months: Months that must elapse before this condition is
            covered.
    """

    claimed_amount: float
    non_payable_amount: float = 0.0
    room_rent_per_day: float | None = None
    hospitalisation_days: int | None = None
    policy_age_months: int | None = None
    exempt_from_proportion: float = 0.0

    sum_insured: PolicyTerm | None = None
    co_pay_percent: PolicyTerm | None = None
    room_rent_cap_per_day: PolicyTerm | None = None
    sub_limit: PolicyTerm | None = None
    deductible: PolicyTerm | None = None
    waiting_period_months: PolicyTerm | None = None

    def __post_init__(self) -> None:
        """Validate the bill facts.

        Raises:
            InvalidClaimInput: If any amount is negative, the co-payment is not
                a percentage, or non-payables exceed the bill.
        """
        if self.claimed_amount < 0:
            raise InvalidClaimInput(f"claimed_amount is negative: {self.claimed_amount}")
        if self.non_payable_amount < 0:
            raise InvalidClaimInput(f"non_payable_amount is negative: {self.non_payable_amount}")
        if self.non_payable_amount > self.claimed_amount:
            raise InvalidClaimInput(
                f"non_payable_amount ({self.non_payable_amount}) exceeds the bill "
                f"({self.claimed_amount})."
            )
        if self.co_pay_percent is not None and not 0 <= self.co_pay_percent.value <= 100:
            raise InvalidClaimInput(
                f"co_pay_percent must be 0-100, got {self.co_pay_percent.value}"
            )


@dataclass
class Step:
    """One rule applied, and what it did to the running balance.

    Attributes:
        rule: Short identifier, e.g. "room_rent_proportion".
        description: Plain-language explanation for the user.
        amount_before: Balance entering this step.
        amount_after: Balance leaving it.
        source_page: Page the governing term came from, when there was one.
    """

    rule: str
    description: str
    amount_before: float
    amount_after: float
    source_page: int | None = None

    @property
    def reduction(self) -> float:
        """Return how much this step removed from the balance."""
        return round(self.amount_before - self.amount_after, ROUNDING)


@dataclass
class ClaimResult:
    """The settled figure plus the full audit trail that produced it.

    Attributes:
        payable: Final amount the insurer pays, in rupees.
        claimed: The original bill, for the headline comparison.
        steps: Every rule applied, in order.
        eligible: False when a waiting period blocks the claim entirely.
        rejection_reason: Why, when `eligible` is False.
        assumptions: Policy terms that were not supplied and therefore not
            applied. Non-empty means this figure is provisional.
        complete: True only when no assumption was made.
        cited_pages: Distinct pages backing the terms used, in first-use order.
    """

    payable: float
    claimed: float
    steps: list[Step] = field(default_factory=list)
    eligible: bool = True
    rejection_reason: str | None = None
    assumptions: list[str] = field(default_factory=list)
    complete: bool = True
    cited_pages: list[int] = field(default_factory=list)

    def explain(self) -> str:
        """Render the breakdown as text.

        Returns:
            A line per rule with its reduction and citation, ending in the
            payable figure — the shape a settlement letter takes, and directly
            usable as an answer.
        """
        lines = [f"Claimed: ₹{self.claimed:,.0f}"]
        if not self.eligible:
            lines.append(f"NOT PAYABLE — {self.rejection_reason}")
            return "\n".join(lines)

        for step in self.steps:
            citation = f" [p.{step.source_page}]" if step.source_page else ""
            if step.reduction:
                lines.append(
                    f"  −₹{step.reduction:,.0f}  {step.description}{citation}"
                )
            else:
                lines.append(f"   (no effect)  {step.description}{citation}")

        lines.append(f"Payable: ₹{self.payable:,.0f}")
        if self.assumptions:
            lines.append("")
            lines.append("PROVISIONAL — the following were not found in the policy:")
            lines.extend(f"  - {assumption}" for assumption in self.assumptions)
        return "\n".join(lines)


def _record(
    result: ClaimResult,
    rule: str,
    description: str,
    before: float,
    after: float,
    page: int | None,
) -> float:
    """Append a step and return the new balance.

    Args:
        result: Result being accumulated.
        rule: Rule identifier.
        description: Plain-language explanation.
        before: Balance entering the step.
        after: Balance leaving it.
        page: Source page of the governing term, if any.

    Returns:
        The rounded balance after this step.
    """
    after = round(after, ROUNDING)
    result.steps.append(
        Step(
            rule=rule,
            description=description,
            amount_before=round(before, ROUNDING),
            amount_after=after,
            source_page=page,
        )
    )
    if page is not None and page not in result.cited_pages:
        result.cited_pages.append(page)
    return after


def settle(request: ClaimRequest) -> ClaimResult:
    """Compute the payable amount, recording every rule that fired.

    The sequence is fixed and documented in the module docstring. It is not
    configurable, deliberately: the order IS the domain logic, and a per-call
    ordering parameter would make two runs of the same claim incomparable.
    Policy-specific *values* vary; the sequence does not.

    Args:
        request: Bill facts plus whichever policy terms were found.

    Returns:
        The settled figure with its full breakdown.
    """
    result = ClaimResult(payable=0.0, claimed=round(request.claimed_amount, ROUNDING))
    balance = float(request.claimed_amount)

    # 1. Waiting period. Checked first because it is binary: if the condition is
    #    not yet covered, no later rule can make the claim payable, and running
    #    them anyway would produce a breakdown implying it nearly was.
    if request.waiting_period_months is not None:
        term = request.waiting_period_months
        if request.policy_age_months is None:
            result.assumptions.append(
                f"Policy age unknown, so the {term.value:.0f}-month waiting period "
                f"[p.{term.source_page}] could not be checked."
            )
        elif request.policy_age_months < term.value:
            result.eligible = False
            result.rejection_reason = (
                f"{term.label}: {term.value:.0f} months required, policy is "
                f"{request.policy_age_months} months old [p.{term.source_page}]"
            )
            result.payable = 0.0
            result.complete = not result.assumptions
            if term.source_page not in result.cited_pages:
                result.cited_pages.append(term.source_page)
            return result
        else:
            # Recorded even though it costs nothing. A check that passed is
            # evidence: it tells the user eligibility was verified rather than
            # skipped, and it puts the governing page in `cited_pages` so the
            # answer can point at the clause that permitted the claim.
            _record(
                result,
                "waiting_period",
                (
                    f"{term.label}: {term.value:.0f} months required, policy is "
                    f"{request.policy_age_months} months old — satisfied"
                ),
                balance,
                balance,
                term.source_page,
            )
    else:
        result.assumptions.append("No waiting period found; assumed none applies.")

    # 2. Non-payables. Removed before any proportioning or capping, because they
    #    were never part of the covered claim to begin with.
    if request.non_payable_amount:
        balance = _record(
            result,
            "non_payable",
            "Non-payable items (consumables, administrative charges)",
            balance,
            balance - request.non_payable_amount,
            None,
        )

    # 3. Room-rent proportionate deduction — the expensive surprise.
    if request.room_rent_cap_per_day is not None:
        cap = request.room_rent_cap_per_day
        if request.room_rent_per_day is None:
            result.assumptions.append(
                f"Actual room rent unknown, so the ₹{cap.value:,.0f}/day limit "
                f"[p.{cap.source_page}] could not be applied."
            )
        elif request.room_rent_per_day > cap.value > 0:
            ratio = cap.value / request.room_rent_per_day
            # Charges the wording exempts are held aside, proportioned nothing,
            # and added back — rather than scaling the whole balance and hoping.
            exempt = min(request.exempt_from_proportion, balance)
            proportionable = balance - exempt
            balance = _record(
                result,
                "room_rent_proportion",
                (
                    f"Room rent ₹{request.room_rent_per_day:,.0f}/day exceeds the "
                    f"₹{cap.value:,.0f}/day limit, so associated charges are paid at "
                    f"{ratio:.1%}"
                ),
                balance,
                proportionable * ratio + exempt,
                cap.source_page,
            )
    else:
        result.assumptions.append("No room-rent limit found; assumed no proportionate deduction.")

    # 4. Sub-limit — a ceiling on this procedure, applied before the deductible
    #    and co-payment because it caps the covered claim, not the settlement.
    if request.sub_limit is not None:
        term = request.sub_limit
        if balance > term.value:
            balance = _record(
                result,
                "sub_limit",
                f"{term.label} caps this claim at ₹{term.value:,.0f}",
                balance,
                term.value,
                term.source_page,
            )
        else:
            _record(
                result,
                "sub_limit",
                f"{term.label} (₹{term.value:,.0f}) not reached",
                balance,
                balance,
                term.source_page,
            )
    else:
        result.assumptions.append("No procedure sub-limit found; assumed none applies.")

    # 5. Deductible.
    if request.deductible is not None:
        term = request.deductible
        balance = _record(
            result,
            "deductible",
            f"{term.label} of ₹{term.value:,.0f} borne by you",
            balance,
            max(0.0, balance - term.value),
            term.source_page,
        )
    else:
        result.assumptions.append("No deductible found; assumed none applies.")

    # 6. Co-payment — a share of what remains, so it must come after every
    #    reduction above. Applying it earlier understates the deduction.
    if request.co_pay_percent is not None:
        term = request.co_pay_percent
        balance = _record(
            result,
            "co_payment",
            f"{term.label}: you bear {term.value:.0f}% of the balance",
            balance,
            balance * (1 - term.value / 100.0),
            term.source_page,
        )
    else:
        result.assumptions.append("No co-payment found; assumed none applies.")

    # 7. Sum insured — the last ceiling, since nothing can exceed it.
    if request.sum_insured is not None:
        term = request.sum_insured
        if balance > term.value:
            balance = _record(
                result,
                "sum_insured",
                f"{term.label} caps payment at ₹{term.value:,.0f}",
                balance,
                term.value,
                term.source_page,
            )
    else:
        result.assumptions.append("Sum insured not found; no overall ceiling applied.")

    result.payable = round(max(0.0, balance), ROUNDING)
    result.complete = not result.assumptions
    return result


# --- Worked examples, used as regression tests -------------------------------
#
# Expected values are computed by hand in the comments so a failure says which
# RULE broke, not merely that a total moved. pytest is not a project dependency,
# and these run in milliseconds with no fixtures.

def _self_test() -> list[tuple[str, bool, str]]:
    """Run the worked examples.

    Returns:
        `(name, passed, detail)` per case.
    """
    cases: list[tuple[str, bool, str]] = []

    def check(name: str, actual: float, expected: float) -> None:
        ok = abs(actual - expected) < 1.0
        cases.append((name, ok, f"expected ₹{expected:,.0f}, got ₹{actual:,.0f}"))

    # Case 1 — the co-payment everyone expects.
    # 240000 * 0.9 = 216000
    result = settle(
        ClaimRequest(
            claimed_amount=240_000,
            co_pay_percent=PolicyTerm(10.0, 39, "Co-payment"),
            sum_insured=PolicyTerm(500_000, 12, "Sum insured"),
            room_rent_cap_per_day=PolicyTerm(0.0, 20, "Room rent limit"),
            sub_limit=None,
            deductible=PolicyTerm(0.0, 14, "Deductible"),
            waiting_period_months=PolicyTerm(0.0, 18, "Waiting period"),
            policy_age_months=36,
        )
    )
    check("co-payment only", result.payable, 216_000)

    # Case 2 — room-rent proportionate deduction, the expensive surprise.
    # Bill 240000, non-payable 20000 -> 220000
    # Room 6000 vs cap 4000 -> ratio 2/3 -> 220000 * 2/3 = 146666.67
    # Co-pay 10% -> 132000
    result = settle(
        ClaimRequest(
            claimed_amount=240_000,
            non_payable_amount=20_000,
            room_rent_per_day=6_000,
            room_rent_cap_per_day=PolicyTerm(4_000, 20, "Room rent limit"),
            co_pay_percent=PolicyTerm(10.0, 39, "Co-payment"),
            sum_insured=PolicyTerm(500_000, 12, "Sum insured"),
            deductible=PolicyTerm(0.0, 14, "Deductible"),
            waiting_period_months=PolicyTerm(0.0, 18, "Waiting period"),
            policy_age_months=36,
        )
    )
    check("room-rent proportion + co-pay", result.payable, 132_000)

    # Case 3 — ORDER MATTERS. Same inputs as case 2 but co-pay applied first
    # would give 240000-20000=220000, *0.9=198000, *2/3=132000 — identical here
    # because both are multiplicative. The order shows up once a FIXED amount
    # (deductible) sits between them:
    # 220000 * 2/3 = 146666.67; -10000 deductible = 136666.67; *0.9 = 123000
    # Co-pay first would give: 220000*0.9=198000; -10000=188000; *2/3=125333.
    result = settle(
        ClaimRequest(
            claimed_amount=240_000,
            non_payable_amount=20_000,
            room_rent_per_day=6_000,
            room_rent_cap_per_day=PolicyTerm(4_000, 20, "Room rent limit"),
            deductible=PolicyTerm(10_000, 14, "Deductible"),
            co_pay_percent=PolicyTerm(10.0, 39, "Co-payment"),
            sum_insured=PolicyTerm(500_000, 12, "Sum insured"),
            waiting_period_months=PolicyTerm(0.0, 18, "Waiting period"),
            policy_age_months=36,
        )
    )
    check("order: proportion -> deductible -> co-pay", result.payable, 123_000)

    # Case 4 — waiting period blocks everything.
    result = settle(
        ClaimRequest(
            claimed_amount=240_000,
            waiting_period_months=PolicyTerm(24.0, 18, "Knee replacement waiting period"),
            policy_age_months=18,
            co_pay_percent=PolicyTerm(10.0, 39, "Co-payment"),
        )
    )
    check("waiting period not served", result.payable, 0)
    cases.append(
        ("waiting period marks ineligible", not result.eligible, str(result.rejection_reason))
    )

    # Case 5 — sub-limit binds below the bill.
    # 150000 -> sub-limit 40000 -> co-pay 10% -> 36000
    result = settle(
        ClaimRequest(
            claimed_amount=150_000,
            sub_limit=PolicyTerm(40_000, 31, "Cataract sub-limit"),
            co_pay_percent=PolicyTerm(10.0, 39, "Co-payment"),
            sum_insured=PolicyTerm(500_000, 12, "Sum insured"),
            room_rent_cap_per_day=PolicyTerm(0.0, 20, "Room rent limit"),
            deductible=PolicyTerm(0.0, 14, "Deductible"),
            waiting_period_months=PolicyTerm(0.0, 18, "Waiting period"),
            policy_age_months=36,
        )
    )
    check("sub-limit binds", result.payable, 36_000)

    # Case 6 — sum insured is the final ceiling.
    # 900000, no other rules -> capped at 500000
    result = settle(
        ClaimRequest(
            claimed_amount=900_000,
            sum_insured=PolicyTerm(500_000, 12, "Sum insured"),
            co_pay_percent=PolicyTerm(0.0, 39, "Co-payment"),
            room_rent_cap_per_day=PolicyTerm(0.0, 20, "Room rent limit"),
            deductible=PolicyTerm(0.0, 14, "Deductible"),
            waiting_period_months=PolicyTerm(0.0, 18, "Waiting period"),
            policy_age_months=36,
        )
    )
    check("sum insured caps", result.payable, 500_000)

    # Case 7 — absent terms are recorded, never silently zeroed.
    result = settle(ClaimRequest(claimed_amount=100_000))
    cases.append(
        (
            "missing terms flagged incomplete",
            not result.complete and len(result.assumptions) >= 5,
            f"{len(result.assumptions)} assumptions recorded",
        )
    )

    # Case 8 — malformed input raises rather than returning a plausible number.
    try:
        settle(ClaimRequest(claimed_amount=-1))
        cases.append(("negative bill rejected", False, "no exception raised"))
    except InvalidClaimInput:
        cases.append(("negative bill rejected", True, "InvalidClaimInput raised"))

    # Case 9 — a term without a usable page is refused at construction.
    try:
        PolicyTerm(10.0, 0, "Co-payment")
        cases.append(("uncitable term rejected", False, "no exception raised"))
    except InvalidClaimInput:
        cases.append(("uncitable term rejected", True, "InvalidClaimInput raised"))

    return cases


def _demo() -> None:
    """Print a worked settlement, showing the breakdown a user would see."""
    request = ClaimRequest(
        claimed_amount=240_000,
        non_payable_amount=20_000,
        room_rent_per_day=6_000,
        hospitalisation_days=4,
        policy_age_months=36,
        room_rent_cap_per_day=PolicyTerm(4_000, 20, "Room rent limit"),
        deductible=PolicyTerm(10_000, 14, "Deductible"),
        co_pay_percent=PolicyTerm(10.0, 39, "Co-payment"),
        sum_insured=PolicyTerm(500_000, 12, "Sum insured"),
        waiting_period_months=PolicyTerm(24.0, 18, "Waiting period"),
    )
    print("\n=== WORKED SETTLEMENT ===")
    print(settle(request).explain())
    print("\nThe room-rent line is the one users never predict: a ₹2,000/day")
    print("upgrade removed ₹73,333 from this claim, far more than the co-payment.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase3_agents.claims_calculator",
        description="Deterministic claim settlement — no LLM involved.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true", help="Run the worked examples.")
    parser.add_argument("--demo", action="store_true", help="Print one worked settlement.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the self-test and/or the demo.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 if everything passed, 1 if any case failed.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.self_test and not args.demo:
        args.self_test = True

    exit_code = 0
    if args.self_test:
        print("\n=== SELF-TEST ===")
        failures = 0
        for name, passed, detail in _self_test():
            marker = "PASS" if passed else "FAIL"
            print(f"  [{marker}] {name:<45} {detail}")
            failures += not passed
        print(f"\n{failures} failure(s)")
        exit_code = 1 if failures else 0

    if args.demo:
        _demo()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
