"""Build the 50-task agent evaluation set, with every fact checked against the corpus.

**What this set is for.** CLAUDE.md Phase 3 task 6 needs "task completion + tool-call
accuracy on a 50-task set", and the phase exit criterion is that the agent beats plain
RAG on *calculation and comparison* tasks. Neither is measurable today: `golden.jsonl`
is 77 `lookup` items plus 15 negatives, so D-22's router eval covered 2 of 4 routes and
had to say so. This file supplies the missing two.

**Why authored, not generated.** The Phase 1 golden set was LLM-generated then filtered,
which suits lookup questions — there are hundreds of equally good ways to ask what the
room-rent rule says. Calculation tasks are different: each one needs a *provably correct
rupee figure*, and a generated arithmetic question is exactly as likely to be wrong as
the model that wrote it. An evals team at an insurer would call this the difference
between a sampled test set and a regression suite. These 50 are a regression suite.

**Why the expected figures are literals rather than `settle()` output.** Ground truth
produced by the tool under test cannot detect that tool's bugs — the suite would agree
with the calculator by construction, including when both are wrong. So every
calculation task carries a hand-computed `expected_payable`, and `verify_arithmetic()`
runs `settle()` and fails the build on mismatch. The literal is the ground truth; the
calculator is the thing being checked.

**Why every citation is machine-verified.** `verify_grounding()` asserts the cited page
actually contains the governing text before a task is written out. A task whose
"correct page" is wrong silently converts a working retriever into a failing one, and
that error is invisible in every downstream metric. Fifty hand-written page numbers
would otherwise be fifty chances to record a wrong answer as ground truth.

**The corpus holds policy WORDINGS, not policy SCHEDULES — and that shapes the set.**
Sum insured, room-rent limits and voluntary co-payment percentages are all "as specified
in the Policy Schedule", a document this project does not have. Fabricating them would
make the grounding fictional. So calculation tasks state the user's plan parameters in
the question (`given` below) and require retrieval to supply the *rule* with its page.
That is also the real product shape: a user knows their sum insured; they do not know
that a ₹6,000 room scales down their surgeon's fee.

Usage:
    python -m evals.build_agent_tasks --help
    python -m evals.build_agent_tasks --verify
    python -m evals.build_agent_tasks --write
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase3_agents.claims_calculator import ClaimRequest, PolicyTerm, settle

LOGGER = logging.getLogger("claimwise.build_agent_tasks")

# Short keys for the four corpus documents, so a task reads as a sentence rather
# than as a hash. Mapped to real doc_ids at build time and verified to exist.
DOCS: dict[str, str] = {
    "star": "starhealth__health__comprehensive.pdf",
    "sbih": "sbigeneral__health__alpha.pdf",
    "home": "sbigeneral__home__house-insurance.pdf",
    "life": "iciciprulife__life__prusmart.pdf",
}

# Route mix, and the reasoning behind it. NOT the natural traffic distribution —
# a real assistant is overwhelmingly `lookup`, which is why "always answer lookup"
# scored 0.837 in D-22 and beat the router.
#
# `lookup` is deliberately UNDER-weighted here because 92 lookup items already
# exist in golden.jsonl; repeating that coverage would add nothing and would let a
# constant classifier score well again. `calculation` and `comparison` are
# over-weighted because they have no labels anywhere in the project and they are
# the two routes the phase exit criterion is stated in terms of.
#
# The consequence must be quoted with any accuracy figure from this set: the
# majority-class baseline here is 14/50 = 0.28, not 0.837, so the two numbers are
# not comparable and neither replaces the other.
TARGET_MIX: dict[str, int] = {
    "lookup": 14,
    "calculation": 14,
    "comparison": 12,
    "out_of_scope": 10,
}


# --- Task definitions --------------------------------------------------------
#
# `evidence` is (doc key, page, phrase that must literally appear on that page).
# The phrase is the machine-checkable part of the claim "this page answers this
# question", and it is checked against data/processed/pages.jsonl at build time.
#
# `given` holds the plan parameters a user would know and the corpus does not
# contain. `expected_payable` is hand-computed; the arithmetic is written out in
# `working` so a failing build says which step disagrees rather than only that a
# total moved.

LOOKUP_TASKS: list[dict[str, Any]] = [
    {
        "id": "t-001",
        "question": "I'm 63 and just took out this Star policy. Do I have to pay part of every claim myself?",
        "expect": "Yes — 10% co-payment on each and every claim for entry age 61 and above.",
        "evidence": [("star", 39, "co-payment of 10% of each and every claim")],
    },
    {
        "id": "t-002",
        "question": "I have diabetes already. How long until my Star policy covers it?",
        "expect": "36 months of continuous coverage.",
        "evidence": [("star", 31, "36 months of continuous coverage")],
    },
    {
        "id": "t-003",
        "question": "I bought the Star policy three weeks ago and was admitted with a fever. Is that covered?",
        "expect": "No — a 30-day initial waiting period applies, except for accidents.",
        "evidence": [("star", 32, "within 30 days from the first policy commencement date")],
    },
    {
        "id": "t-004",
        "question": "Can I pay more to shorten the pre-existing disease wait on my Star policy?",
        "expect": "Yes — an optional buy-back reduces it from 36 months to 12 months.",
        "evidence": [("star", 30, "from 36 months to 12 months")],
    },
    {
        "id": "t-005",
        "question": "How much will Star pay towards an air ambulance?",
        "expect": "Up to Rs.2,50,000 per hospitalisation, capped at Rs.5,00,000 per policy period.",
        "evidence": [("star", 11, "air ambulance service up to Rs.2,50,000")],
    },
    {
        "id": "t-006",
        "question": "Does the Star policy pay for treatment I receive at home instead of in hospital?",
        "expect": "Yes — home care treatment up to 10% of sum insured, maximum Rs.5 lakhs per policy year.",
        "evidence": [("star", 11, "Home Care Treatment")],
    },
    {
        "id": "t-007",
        "question": "What's the waiting period for pre-existing conditions on my SBI Alpha policy?",
        "expect": "24 months of continuous coverage.",
        "evidence": [("sbih", 20, "24 months of continuo")],
    },
    {
        "id": "t-008",
        "question": "When can I claim maternity expenses under SBI Alpha?",
        "expect": "After a 24-month waiting period.",
        "evidence": [("sbih", 21, "Maternity and Child Care Cover")],
    },
    {
        "id": "t-009",
        "question": "Is there a waiting period before the SBI Alpha critical illness cover starts?",
        "expect": "Yes — 90 days from policy commencement.",
        "evidence": [("sbih", 17, "Waiting Period of 90 days")],
    },
    {
        "id": "t-010",
        "question": "Can the specific-disease waiting period on SBI Alpha be shortened?",
        "expect": "Yes — an optional benefit reduces the 24-month specific-disease wait to 12 months.",
        "evidence": [("sbih", 19, "to 12 months")],
    },
    {
        "id": "t-011",
        "question": "My three-year-old laptop was stolen. What depreciation will you apply to the claim?",
        "expect": "20% — the rate for items up to 3 years old.",
        "evidence": [("home", 5, "Up to 3 Years 20%")],
    },
    {
        "id": "t-012",
        "question": "If I cancel my home insurance two months in, how much premium do I get back?",
        "expect": "50% of the annual rate, for a policy in force up to three months.",
        "evidence": [("home", 6, "Up to three months")],
    },
    {
        "id": "t-013",
        "question": "I'm building an extension on my house. Do I need to tell the insurer?",
        "expect": "Yes, if it increases carpet area by more than 10%; additional premium must be paid.",
        "evidence": [("home", 7, "10% of the Carpet Area")],
    },
    {
        "id": "t-014",
        "question": "Can I withdraw money from my ICICI ULIP part-way through, and is there a fee?",
        "expect": "Yes — partial withdrawals up to 20% of fund value per year, minimum Rs.2,000, no charge.",
        "evidence": [("life", 3, "20% of the Fund Value")],
    },
]

CALCULATION_TASKS: list[dict[str, Any]] = [
    {
        "id": "t-015",
        # CLAUDE.md's own motivating question, minus the waiting-period half.
        "question": (
            "I'm 63, on the Star Comprehensive policy, and my hospital bill is Rs.2,40,000. "
            "What will I actually get back?"
        ),
        "given": {"claimed_amount": 240000},
        "terms": {"co_pay_percent": (10.0, 39, "Co-payment for entry age 61+")},
        "expected_payable": 216000,
        "working": "240,000 - 10% co-pay = 216,000",
        "expected_complete": False,
        "evidence": [("star", 39, "co-payment of 10% of each and every claim")],
    },
    {
        "id": "t-016",
        "question": (
            "Star policy, I'm 65, sum insured Rs.5,00,000, and the bill came to Rs.6,00,000. "
            "How much do I receive?"
        ),
        "given": {"claimed_amount": 600000},
        "terms": {
            "co_pay_percent": (10.0, 39, "Co-payment for entry age 61+"),
            "sum_insured": (500000.0, 9, "Sum insured"),
        },
        "expected_payable": 500000,
        "working": "600,000 - 10% = 540,000, capped at the 500,000 sum insured",
        "expected_complete": False,
        "evidence": [("star", 39, "co-payment of 10% of each and every claim")],
    },
    {
        "id": "t-017",
        "question": (
            "My SBI Alpha room limit is Rs.4,000 a day but I stayed in a Rs.6,000 room. "
            "The total bill is Rs.3,00,000. What gets paid?"
        ),
        "given": {"claimed_amount": 300000, "room_rent_per_day": 6000, "hospitalisation_days": 4},
        "terms": {"room_rent_cap_per_day": (4000.0, 27, "Eligible room rent limit")},
        "expected_payable": 200000,
        "working": "4,000/6,000 = 66.67%; 300,000 x 2/3 = 200,000",
        "expected_complete": False,
        "evidence": [("sbih", 27, "Eligible Room Rent limit / Room Rent actually incurred")],
    },
    {
        "id": "t-018",
        # The order-of-operations case. A deductible is subtractive and a
        # co-payment multiplicative, so swapping them changes the answer —
        # unlike two multiplicative steps, which commute and would prove nothing.
        "question": (
            "Bill of Rs.3,00,000, my policy has a Rs.50,000 deductible and a 10% co-payment. "
            "What do I get?"
        ),
        "given": {"claimed_amount": 300000},
        "terms": {
            "deductible": (50000.0, 20, "Voluntary deductible"),
            "co_pay_percent": (10.0, 20, "Voluntary co-payment"),
        },
        "expected_payable": 225000,
        "working": "(300,000 - 50,000) x 0.9 = 225,000. Co-pay first would give 220,000 — wrong.",
        "expected_complete": False,
        "evidence": [("sbih", 20, "Voluntary Co-Payment")],
    },
    {
        "id": "t-019",
        # CLAUDE.md's motivating question in full: knee surgery at 18 months.
        "question": (
            "My Star policy is 18 months old and I need knee surgery for arthritis I already had. "
            "The estimate is Rs.2,40,000. What will I get back?"
        ),
        "given": {"claimed_amount": 240000, "policy_age_months": 18},
        "terms": {"waiting_period_months": (36.0, 31, "Pre-existing disease waiting period")},
        "expected_payable": 0,
        "working": "36-month PED wait not served at 18 months — nothing payable",
        "expected_eligible": False,
        # complete=True even though nothing is payable, and that is correct:
        # `complete` means "no term had to be assumed", not "the claim succeeded".
        # The waiting-period check short-circuits before any other term is
        # consulted, so nothing was guessed. The build caught this expectation
        # being wrong, which is the check working in the intended direction.
        "expected_complete": True,
        "evidence": [("star", 31, "36 months of continuous coverage")],
    },
    {
        "id": "t-020",
        "question": (
            "Same knee surgery, but my Star policy has now run 48 months. Bill Rs.2,40,000, "
            "and I'm 63. What do I get?"
        ),
        "given": {"claimed_amount": 240000, "policy_age_months": 48},
        "terms": {
            "waiting_period_months": (36.0, 31, "Pre-existing disease waiting period"),
            "co_pay_percent": (10.0, 39, "Co-payment for entry age 61+"),
        },
        "expected_payable": 216000,
        "working": "36-month wait served at 48 months; 240,000 x 0.9 = 216,000",
        "expected_complete": False,
        "evidence": [
            ("star", 31, "36 months of continuous coverage"),
            ("star", 39, "co-payment of 10% of each and every claim"),
        ],
    },
    {
        "id": "t-021",
        "question": (
            "Bill Rs.2,40,000 and the hospital listed Rs.18,000 of gloves, syringes and admin "
            "charges. I'm 63 on the Star policy. What's payable?"
        ),
        "given": {"claimed_amount": 240000, "non_payable_amount": 18000},
        "terms": {"co_pay_percent": (10.0, 39, "Co-payment for entry age 61+")},
        "expected_payable": 199800,
        "working": "(240,000 - 18,000) x 0.9 = 199,800",
        "expected_complete": False,
        "evidence": [("star", 39, "co-payment of 10% of each and every claim")],
    },
    {
        "id": "t-022",
        "question": (
            "I had a modern treatment procedure capped at Rs.1,25,000 under my Star policy. "
            "The bill was Rs.3,00,000 and I'm 63. What do I receive?"
        ),
        "given": {"claimed_amount": 300000},
        "terms": {
            "sub_limit": (125000.0, 10, "Modern treatment sub-limit"),
            "co_pay_percent": (10.0, 39, "Co-payment for entry age 61+"),
        },
        "expected_payable": 112500,
        "working": "capped at 125,000, then x 0.9 = 112,500",
        "expected_complete": False,
        "evidence": [("star", 10, "1,25,000")],
    },
    {
        "id": "t-023",
        # Every rule firing at once, which is where an LLM doing this in its head
        # reliably diverges.
        "question": (
            "Star policy, I'm 63, policy 48 months old. Bill Rs.4,50,000 including Rs.30,000 "
            "of consumables. I used a Rs.5,000/day room against a Rs.4,000 limit, the procedure "
            "is capped at Rs.3,00,000, there's a Rs.25,000 deductible and my sum insured is "
            "Rs.5,00,000. What is payable?"
        ),
        "given": {
            "claimed_amount": 450000,
            "non_payable_amount": 30000,
            "room_rent_per_day": 5000,
            "hospitalisation_days": 5,
            "policy_age_months": 48,
        },
        "terms": {
            "waiting_period_months": (36.0, 31, "Pre-existing disease waiting period"),
            "room_rent_cap_per_day": (4000.0, 9, "Eligible room rent limit"),
            "sub_limit": (300000.0, 10, "Procedure sub-limit"),
            "deductible": (25000.0, 20, "Deductible"),
            "co_pay_percent": (10.0, 39, "Co-payment for entry age 61+"),
            "sum_insured": (500000.0, 9, "Sum insured"),
        },
        "expected_payable": 247500,
        "working": (
            "450,000 - 30,000 = 420,000; x 4/5 room ratio = 336,000; sub-limit 300,000 not "
            "exceeded? it is: 336,000 -> 300,000; -25,000 = 275,000; x 0.9 = 247,500; under SI"
        ),
        "expected_complete": True,
        "evidence": [("star", 9, "in proportion to the room rent limit")],
    },
    {
        "id": "t-024",
        # Pharmacy exempted from proportioning — the wording varies by insurer,
        # which is why the calculator takes it as an input rather than assuming.
        "question": (
            "Bill Rs.3,00,000 of which Rs.50,000 is pharmacy, which my policy exempts from "
            "proportionate deduction. Room was Rs.6,000/day against a Rs.4,000 limit. "
            "What's payable?"
        ),
        "given": {
            "claimed_amount": 300000,
            "room_rent_per_day": 6000,
            "hospitalisation_days": 3,
            "exempt_from_proportion": 50000,
        },
        "terms": {"room_rent_cap_per_day": (4000.0, 27, "Eligible room rent limit")},
        "expected_payable": 216667,
        "working": "(300,000 - 50,000) x 2/3 = 166,666.67, + 50,000 exempt = 216,666.67 -> 216,667",
        "expected_complete": False,
        "evidence": [("sbih", 27, "Eligible Room Rent limit / Room Rent actually incurred")],
    },
    {
        "id": "t-025",
        # The "absent is not zero" case. SBI's co-payment percentage lives in the
        # Policy Schedule, so a correct agent cannot produce a final figure — it
        # must return a provisional one and say what is missing. An agent that
        # answers 2,40,000 flat has silently assumed no co-payment.
        "question": (
            "My SBI Alpha policy has voluntary co-payment. Bill is Rs.2,40,000. "
            "What will I get back?"
        ),
        "given": {"claimed_amount": 240000},
        "terms": {},
        "expected_payable": 240000,
        "working": "no terms found — 240,000 provisional, every term recorded as an assumption",
        "expected_complete": False,
        "requires_assumption": True,
        "evidence": [("sbih", 20, "Voluntary Co-Payment")],
    },
    {
        "id": "t-026",
        "question": "Bill of Rs.1,00,000 against a Rs.25,000 voluntary deductible. What's payable?",
        "given": {"claimed_amount": 100000},
        "terms": {"deductible": (25000.0, 20, "Voluntary deductible")},
        "expected_payable": 75000,
        "working": "100,000 - 25,000 = 75,000",
        "expected_complete": False,
        "evidence": [("sbih", 20, "Voluntary Deductible")],
    },
    {
        "id": "t-027",
        # Flagged: the calculator has no depreciation rule, so this is a real user
        # question the current tool cannot settle. Recorded rather than dropped —
        # a task set that only contains what the tools already do measures nothing
        # about coverage.
        "question": (
            "My three-year-old laptop worth Rs.80,000 was stolen. After depreciation, "
            "what does the home policy pay?"
        ),
        "given": {"claimed_amount": 80000},
        "terms": {},
        "expected_payable": 64000,
        "working": "80,000 - 20% depreciation (up to 3 years) = 64,000",
        "expected_complete": False,
        "requires_unimplemented": "depreciation_schedule",
        "evidence": [("home", 5, "Up to 3 Years 20%")],
    },
    {
        "id": "t-028",
        # Flagged for the same reason: a ULIP withdrawal cap is a different
        # calculation from a hospitalisation settlement.
        "question": (
            "My ULIP fund value is Rs.8,00,000 and I want to withdraw Rs.2,00,000 this year. "
            "How much can I actually take out?"
        ),
        "given": {"claimed_amount": 200000},
        "terms": {},
        "expected_payable": 160000,
        "working": "20% of 8,00,000 fund value = 160,000 maximum in a policy year",
        "expected_complete": False,
        "requires_unimplemented": "ulip_withdrawal_cap",
        "evidence": [("life", 3, "20% of the Fund Value")],
    },
]

COMPARISON_TASKS: list[dict[str, Any]] = [
    {
        "id": "t-029",
        "question": "Which of my two health policies makes me wait less for pre-existing conditions?",
        "expect": "SBI Alpha at 24 months, versus Star at 36 months.",
        "evidence": [
            ("star", 31, "36 months of continuous coverage"),
            ("sbih", 20, "24 months of continuo"),
        ],
    },
    {
        "id": "t-030",
        "question": "Do both my health policies have a 30-day wait when the policy first starts?",
        "expect": "Yes — both apply a 30-day initial waiting period except for accidents.",
        "evidence": [
            ("star", 32, "within 30 days from the first policy commencement date"),
            ("sbih", 21, "within 30 days from the first Policy commencement date"),
        ],
    },
    {
        "id": "t-031",
        "question": "Compare the co-payment I'd bear under my Star policy versus my SBI Alpha policy.",
        "expect": (
            "Star imposes a mandatory 10% for entry age 61+; SBI Alpha's is voluntary and the "
            "percentage sits in the Policy Schedule, which is not in the wording."
        ),
        "evidence": [
            ("star", 39, "co-payment of 10% of each and every claim"),
            ("sbih", 20, "Voluntary Co-Payment"),
        ],
    },
    {
        "id": "t-032",
        "question": "If I take a room above my limit, do both health policies cut my claim the same way?",
        "expect": "Both apply a proportionate deduction to associated medical expenses.",
        "evidence": [
            ("star", 9, "in proportion to the room rent limit"),
            ("sbih", 27, "Eligible Room Rent limit / Room Rent actually incurred"),
        ],
    },
    {
        "id": "t-033",
        "question": "Which policy lets me buy down the waiting period, and by how much?",
        "expect": (
            "Star offers a PED buy-back from 36 to 12 months; SBI Alpha reduces the "
            "specific-disease wait from 24 to 12 months."
        ),
        "evidence": [
            ("star", 30, "from 36 months to 12 months"),
            ("sbih", 19, "to 12 months"),
        ],
    },
    {
        "id": "t-034",
        "question": "How long is a 'specific waiting period' under each of my health policies?",
        "expect": "Star defines it as up to 36 months; SBI Alpha as up to 24 months.",
        "evidence": [
            ("star", 7, "36 months"),
            ("sbih", 3, "24 months"),
        ],
    },
    {
        "id": "t-035",
        "question": (
            "A pipe burst and flooded my flat, and I hurt my back moving furniture. "
            "Which policy covers which part?"
        ),
        "expect": (
            "Property damage under the home policy; the back injury under a health policy. "
            "Two different policies, two different claims."
        ),
        "evidence": [
            ("home", 2, "Sum Insured"),
            ("sbih", 5, "Room rent and boarding expenses"),
        ],
    },
    {
        "id": "t-036",
        "question": "Is my sum insured worked out the same way on my home policy as on my health policy?",
        "expect": (
            "No — the home building sum insured is the cost of construction, while the health "
            "sum insured is the amount stated in the Policy Schedule."
        ),
        "evidence": [
            ("home", 4, "Cost of Construction"),
            ("sbih", 5, "as specified in the Policy Schedule"),
        ],
    },
    {
        "id": "t-037",
        "question": "If I die in an accident, do my life rider and my health policy both pay out?",
        "expect": (
            "They are separate covers with separate conditions; the life accident rider has its "
            "own exclusions and no surrender value."
        ),
        "evidence": [
            ("life", 8, "no surrender value for the rider"),
            ("star", 19, "Accidental Death"),
        ],
    },
    {
        "id": "t-038",
        "question": "Which of my policies pays a cash bonus for not claiming, and how much?",
        "expect": (
            "Star grants a cumulative bonus of 50% of the basic sum insured per claim-free year "
            "at Rs.5,00,000 sum insured; SBI Alpha has a No Claim Bonus section."
        ),
        "evidence": [
            ("star", 12, "Cumulative Bonus"),
            ("sbih", 20, "No Claim Bonus"),
        ],
    },
    {
        "id": "t-039",
        "question": "I'm 63 and choosing between my two health policies for a planned surgery. Which costs me less?",
        "expect": (
            "Star applies a mandatory 10% co-payment at entry age 61+; SBI Alpha's co-payment is "
            "voluntary. The comparison cannot be completed without the SBI Policy Schedule."
        ),
        "evidence": [
            ("star", 39, "co-payment of 10% of each and every claim"),
            ("sbih", 20, "Voluntary Co-Payment"),
        ],
        "requires_assumption": True,
    },
    {
        "id": "t-040",
        "question": "Do my home policy and my health policies share one sum insured between them?",
        "expect": "No — they are separate policies, each with its own sum insured.",
        "evidence": [
            ("home", 2, "Sum Insured"),
            ("star", 9, "Sum Insured"),
        ],
    },
]

# Unanswerable from this corpus. Half are plausible insurance questions about
# lines we simply do not hold (the realistic failure), half are outside insurance
# altogether. Deliberately disjoint from golden.jsonl's 15 negatives so the two
# sets can be reported separately without double-counting.
OUT_OF_SCOPE_TASKS: list[dict[str, Any]] = [
    {"id": "t-041", "question": "What premium would I pay to insure my Honda City?"},
    {"id": "t-042", "question": "Does my travel policy reimburse me if my flight to Dubai is cancelled?"},
    {"id": "t-043", "question": "How do I insure my wheat crop against a failed monsoon?"},
    {"id": "t-044", "question": "What does marine cargo insurance cover for a shipment to Singapore?"},
    {"id": "t-045", "question": "What is the IDV on a two-year-old motorcycle?"},
    {"id": "t-046", "question": "Can you file my income tax return for this financial year?"},
    {"id": "t-047", "question": "How do I apply for a driving licence in Maharashtra?"},
    {"id": "t-048", "question": "Will it rain in Mumbai tomorrow?"},
    {"id": "t-049", "question": "What is SBI's share price today?"},
    {"id": "t-050", "question": "What GST rate applies to restaurant food?"},
]


def load_pages(processed_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Read the parsed corpus, keyed by (filename, page).

    Args:
        processed_dir: Directory holding `pages.jsonl`.

    Returns:
        Page records by filename and page number.

    Raises:
        FileNotFoundError: If the corpus has not been ingested.
    """
    path = processed_dir / "pages.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path.resolve()} not found. data/processed is gitignored; rebuild with "
            "`python -m phase1_rag.ingest`."
        )
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(row["filename"], row["page"]): row for row in rows}


def normalise(text: str) -> str:
    """Collapse whitespace and lowercase, for tolerant phrase matching.

    PDF extraction leaves hard line wraps mid-sentence (P-12), so a phrase that
    reads as one line in the document is not one line in the text. Matching on
    collapsed whitespace is what makes a hand-written phrase checkable at all.

    Args:
        text: Raw page or phrase text.

    Returns:
        Whitespace-collapsed lowercase text.
    """
    return " ".join(text.split()).lower()


def verify_grounding(
    tasks: list[dict[str, Any]], pages: dict[tuple[str, int], dict[str, Any]]
) -> list[str]:
    """Check that every cited page really contains the text it is cited for.

    Args:
        tasks: Task definitions carrying `evidence`.
        pages: Corpus pages by (filename, page).

    Returns:
        One human-readable problem per failed citation; empty when all hold.
    """
    problems: list[str] = []
    for task in tasks:
        for doc_key, page, phrase in task.get("evidence", []):
            filename = DOCS.get(doc_key)
            if filename is None:
                problems.append(f"{task['id']}: unknown document key {doc_key!r}")
                continue
            record = pages.get((filename, page))
            if record is None:
                problems.append(f"{task['id']}: {filename} has no page {page}")
                continue
            if normalise(phrase) not in normalise(record["text"]):
                problems.append(
                    f"{task['id']}: {filename} p.{page} does not contain {phrase!r}"
                )
    return problems


def build_request(task: dict[str, Any]) -> ClaimRequest:
    """Turn a calculation task's `given` and `terms` into a claim request.

    Args:
        task: A calculation task definition.

    Returns:
        The request the calculator should be given.
    """
    terms = {
        name: PolicyTerm(value=value, source_page=page, label=label)
        for name, (value, page, label) in task.get("terms", {}).items()
    }
    return ClaimRequest(**task.get("given", {}), **terms)


def verify_arithmetic(tasks: list[dict[str, Any]]) -> list[str]:
    """Check each hand-computed figure against the calculator.

    The literal in the task is the ground truth and the calculator is what is
    being checked — not the other way round. A mismatch means one of them is
    wrong and the build stops until a human decides which.

    Tasks flagged `requires_unimplemented` are skipped: their correct answer is
    known but the tool cannot express the rule yet, so running `settle()` on them
    would compare against a calculation nobody claims it performs.

    Args:
        tasks: Calculation task definitions.

    Returns:
        One problem per disagreement; empty when all agree.
    """
    problems: list[str] = []
    for task in tasks:
        if task.get("requires_unimplemented"):
            continue
        result = settle(build_request(task))

        if result.payable != task["expected_payable"]:
            problems.append(
                f"{task['id']}: calculator says {result.payable:,.0f}, task says "
                f"{task['expected_payable']:,.0f} ({task['working']})"
            )
        expected_eligible = task.get("expected_eligible", True)
        if result.eligible != expected_eligible:
            problems.append(
                f"{task['id']}: eligible={result.eligible}, expected {expected_eligible}"
            )
        if "expected_complete" in task and result.complete != task["expected_complete"]:
            problems.append(
                f"{task['id']}: complete={result.complete}, expected "
                f"{task['expected_complete']}"
            )
        if task.get("requires_assumption") and not result.assumptions:
            problems.append(f"{task['id']}: expected assumptions to be recorded, got none")
    return problems


def to_record(task: dict[str, Any], route: str, doc_ids: dict[str, str]) -> dict[str, Any]:
    """Render one task definition as an eval-set row.

    Args:
        task: The authored definition.
        route: The route this task belongs to.
        doc_ids: Document key to doc_id.

    Returns:
        A JSON-serialisable task record.
    """
    evidence = [
        {
            "doc_key": doc_key,
            "doc_id": doc_ids[doc_key],
            "filename": DOCS[doc_key],
            "page": page,
            "phrase": phrase,
        }
        for doc_key, page, phrase in task.get("evidence", [])
    ]

    # Tool expectations are derived from the route rather than written per task,
    # so a route's contract cannot drift item by item. `out_of_scope` expects NO
    # retrieval: searching a corpus that cannot contain the answer wastes the
    # call and invites a plausible-looking wrong citation.
    expected_tools = {
        "lookup": ["retrieve"],
        "calculation": ["retrieve", "claims_calculator"],
        "comparison": ["retrieve"],
        "out_of_scope": [],
    }[route]

    record: dict[str, Any] = {
        "id": task["id"],
        "question": task["question"],
        "route": route,
        "expected_tools": expected_tools,
        "evidence": evidence,
        "ground_truth_pages": sorted({item["page"] for item in evidence}),
        "ground_truth_refs": sorted({f"{item['doc_id']}:{item['page']}" for item in evidence}),
        # Comparison is defined by needing more than one document, and that is
        # checkable: an answer citing a single policy has not compared anything.
        "min_documents": 2 if route == "comparison" else (0 if route == "out_of_scope" else 1),
        "should_refuse": route == "out_of_scope",
        "verified": False,
    }

    if "expect" in task:
        record["expected_answer"] = task["expect"]
    if route == "calculation":
        record["given"] = task.get("given", {})
        record["policy_terms"] = {
            name: {"value": value, "source_page": page, "label": label}
            for name, (value, page, label) in task.get("terms", {}).items()
        }
        record["expected_payable"] = task["expected_payable"]
        record["working"] = task["working"]
        record["expected_eligible"] = task.get("expected_eligible", True)
        record["expected_complete"] = task.get("expected_complete", True)
    if task.get("requires_assumption"):
        record["requires_assumption"] = True
    if task.get("requires_unimplemented"):
        record["requires_unimplemented"] = task["requires_unimplemented"]
    return record


def build(pages: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Assemble and verify the whole task set.

    Args:
        pages: Corpus pages by (filename, page).

    Returns:
        The task records and any problems found. A non-empty problem list means
        the set must not be written.
    """
    doc_ids: dict[str, str] = {}
    problems: list[str] = []
    for key, filename in DOCS.items():
        match = next((row for (name, _), row in pages.items() if name == filename), None)
        if match is None:
            problems.append(f"corpus is missing {filename}")
        else:
            doc_ids[key] = match["doc_id"]

    grouped = [
        ("lookup", LOOKUP_TASKS),
        ("calculation", CALCULATION_TASKS),
        ("comparison", COMPARISON_TASKS),
        ("out_of_scope", OUT_OF_SCOPE_TASKS),
    ]

    all_tasks = [task for _, tasks in grouped for task in tasks]
    problems.extend(verify_grounding(all_tasks, pages))
    problems.extend(verify_arithmetic(CALCULATION_TASKS))

    ids = [task["id"] for task in all_tasks]
    duplicates = [task_id for task_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        problems.append(f"duplicate task ids: {duplicates}")

    # P-19 cost this project a re-baseline: eight byte-identical questions sat in
    # the golden set for two phases, silently double-weighting themselves. Check
    # here, where it is free, rather than discovering it in a metric.
    questions = [normalise(task["question"]) for task in all_tasks]
    repeated = [question for question, count in Counter(questions).items() if count > 1]
    if repeated:
        problems.append(f"duplicate questions: {repeated}")

    for route, tasks in grouped:
        if len(tasks) != TARGET_MIX[route]:
            problems.append(
                f"route {route}: {len(tasks)} tasks, target mix says {TARGET_MIX[route]}"
            )

    if problems:
        return [], problems

    records = [
        to_record(task, route, doc_ids) for route, tasks in grouped for task in tasks
    ]
    return records, []


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe the assembled set.

    Args:
        records: Task records.

    Returns:
        Counts used for the build report and written into the metadata file.
    """
    routes = Counter(record["route"] for record in records)
    majority = max(routes.values()) / len(records) if records else 0.0
    documents = Counter(
        item["doc_key"] for record in records for item in record["evidence"]
    )
    return {
        "total": len(records),
        "by_route": dict(routes),
        "majority_class_baseline": round(majority, 4),
        "by_document": dict(documents),
        "calculation_runnable": sum(
            1
            for record in records
            if record["route"] == "calculation" and "requires_unimplemented" not in record
        ),
        "requires_unimplemented": sorted(
            record["id"] for record in records if "requires_unimplemented" in record
        ),
        "requires_assumption": sorted(
            record["id"] for record in records if record.get("requires_assumption")
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.build_agent_tasks",
        description="Build and verify the 50-task agent eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--verify", action="store_true", help="Check grounding and arithmetic, write nothing."
    )
    parser.add_argument("--write", action="store_true", help="Write the task set to data/eval.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Verify and optionally write the agent task set.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if verification failed.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verify and not args.write:
        args.verify = True

    config = load_config(args.config)
    processed_dir = Path(cfg_get(config, "paths.processed_dir", "data/processed"))
    eval_dir = Path(cfg_get(config, "paths.eval_dir", "data/eval"))

    pages = load_pages(processed_dir)
    records, problems = build(pages)

    if problems:
        print(f"\n=== BUILD FAILED — {len(problems)} problem(s) ===")
        for problem in problems:
            print(f"  {problem}")
        print("\nNothing written. Every citation and every figure must hold first.")
        return 1

    summary = summarise(records)
    print("\n=== AGENT TASK SET ===")
    print(f"total                : {summary['total']}")
    for route, count in sorted(summary["by_route"].items()):
        print(f"  {route:<16}: {count}")
    print(f"majority baseline    : {summary['majority_class_baseline']}  <- routers must beat this")
    print(f"calculation runnable : {summary['calculation_runnable']} of {summary['by_route']['calculation']}")
    print(f"needs new tooling    : {summary['requires_unimplemented']}")
    print(f"must flag assumption : {summary['requires_assumption']}")
    print(f"evidence by document : {summary['by_document']}")
    print("\nall citations verified against data/processed/pages.jsonl")
    print("all rupee figures verified against phase3_agents.claims_calculator")

    if args.write:
        eval_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = eval_dir / "agent_tasks.jsonl"
        with tasks_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        meta_path = eval_dir / "agent_tasks.meta.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nwrote {tasks_path}")
        print(f"wrote {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
