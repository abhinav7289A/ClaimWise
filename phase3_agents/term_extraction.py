"""Turn retrieved policy text into calculator inputs — the bridge, and the weak link.

**Why this exists.** `claims_calculator.py` settles a claim from structured
terms: a co-payment percentage, a room-rent cap, a waiting period, each carrying
the page it was read from. Retrieval returns prose. Something has to cross that
gap, and until it does the calculator cannot run inside the graph at all — it can
only be driven by hand-supplied numbers, which is how it has been tested so far.

**Why regular expressions and not an LLM.** CLAUDE.md forbids LLM arithmetic, and
the same reasoning extends one step earlier: a model that misreads "10%" as "20%"
produces a confidently wrong rupee figure that no downstream check catches. A
regex either matches the exact wording or it does not, and when it does not, the
term stays `None` — which the calculator already handles by marking the result
provisional and naming the missing term. That asymmetry is the whole design:

* **False negative** (term missed) → the answer is flagged provisional and says
  which term was not found. Visibly incomplete, and correct about being incomplete.
* **False positive** (term misread) → a specific, confident, wrong rupee amount.

The second failure is far more expensive than the first, so every pattern here
is deliberately narrow. **This module is expected to under-extract.**

**Two sources, precedence by kind of number.** Terms arrive either from
retrieved policy text (`extract_terms`, each carrying a page) or from the user's
own question (`extract_question_terms`, carrying no page). `merge_terms`
combines them, and which one wins depends on what the number *is*:

* A **policy-wide rule** — a co-payment percentage, a waiting period — is stated
  once for the whole wording and is the same for every holder. The document wins.
* A **plan-scoped amount** — sum insured, deductible, sub-limit, room-rent cap —
  appears in the document at several tiers at once. Retrieval cannot tell which
  tier this user bought; the user can. The user wins.

Disagreements are reported either way rather than swallowed.

Bill facts — the amount claimed, non-payable heads, the room actually occupied —
come from the question only; they describe one claim and cannot appear in a
policy document. `build_claim_request` populates **every** `ClaimRequest` field
from these two sources. An earlier version populated five of twelve, which
silently disabled the non-payable, room-rent-proportion and sub-limit rules
however the question was worded.

**This is the least-measured component in Phase 3.** The router, calculator, gate
and retrieval node all have recorded numbers. This one has a self-test and no
eval yet. `agent_tasks.jsonl` carries hand-verified `policy_terms` for its 14
calculation tasks, which is exactly the ground truth an extraction eval needs —
that measurement is the obvious next step and has not been taken.

Usage:
    python -m phase3_agents.term_extraction --help
    python -m phase3_agents.term_extraction --self-test
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from typing import Any

from phase3_agents.claims_calculator import ClaimRequest, PolicyTerm

LOGGER = logging.getLogger("claimwise.term_extraction")

# Indian digit grouping: 2,40,000 rather than 240,000. A pattern written for
# Western grouping silently truncates at the first group and reads 2,40,000 as
# 2,40 — a 1000x error that looks like a plausible number.
_AMOUNT = r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)"
_LAKH = r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b"
_CRORE = r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b"

# Each pattern must see the governing noun and the number close together. The
# window is what stops "36 months" from a renewal clause being read as a
# pre-existing-disease waiting period two paragraphs away.
CO_PAY_PATTERNS = (
    re.compile(r"co-?payment[^.]{0,80}?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*%[^.]{0,40}?co-?payment", re.IGNORECASE),
)

PED_WAITING_PATTERNS = (
    re.compile(
        r"pre-?existing[^.]{0,160}?(\d+)\s*months",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(\d+)\s*months[^.]{0,80}?pre-?existing",
        re.IGNORECASE | re.DOTALL,
    ),
)

ROOM_RENT_CAP_PATTERNS = (
    re.compile(rf"room\s*rent[^.]{{0,120}}?{_AMOUNT}\s*(?:per|/)\s*day", re.IGNORECASE),
    re.compile(rf"{_AMOUNT}\s*(?:per|/)\s*day[^.]{{0,80}}?room\s*rent", re.IGNORECASE),
)

SUM_INSURED_PATTERNS = (
    re.compile(rf"sum\s*insured[^.]{{0,60}}?{_LAKH}", re.IGNORECASE),
    re.compile(rf"sum\s*insured[^.]{{0,60}}?{_AMOUNT}", re.IGNORECASE),
)

DEDUCTIBLE_PATTERNS = (
    re.compile(rf"deductible[^.]{{0,80}}?{_AMOUNT}", re.IGNORECASE),
)

# Document-side only, and deliberately narrow: it requires the word "sub-limit".
# "capped at" is left to the question patterns below, because policy prose caps
# a dozen unrelated things and reading the wrong one produces a confident wrong
# figure — the failure this module's docstring is written around.
SUB_LIMIT_PATTERNS = (
    re.compile(rf"sub-?limit[^.]{{0,60}}?{_AMOUNT}", re.IGNORECASE),
    re.compile(rf"{_AMOUNT}[^.]{{0,30}}?sub-?limit", re.IGNORECASE),
)

# --- Question-only patterns ---------------------------------------------------
#
# Everything below reads the USER's phrasing, not policy prose. They are kept
# separate from the document patterns above because the two texts do not look
# alike: a policy says "Eligible Room Rent limit / Room Rent actually incurred",
# a user says "a Rs.6,000 room against a Rs.4,000 limit". A pattern loose enough
# to catch both would be loose enough to misfire on both.

# The bill must be bound to a governing noun. Taking the first rupee figure in
# the sentence is what read "room limit is Rs.4,000" as a ₹4,000 hospital bill
# and settled t-017 at ₹4,000 against an expected ₹200,000.
_BILL_NOUNS = r"(?:total\s+bill|hospital\s+bill|bill|invoice|estimate|claim(?:ed)?\s+amount|worth)"
BILL_PATTERNS = (
    re.compile(rf"{_BILL_NOUNS}[^.]{{0,25}}?{_AMOUNT}", re.IGNORECASE),
    re.compile(rf"{_BILL_NOUNS}[^.]{{0,25}}?{_LAKH}", re.IGNORECASE),
    re.compile(rf"{_BILL_NOUNS}[^.]{{0,25}}?{_CRORE}", re.IGNORECASE),
)

# Heads the policy never pays. Matched on the CATEGORY noun rather than a list
# of items, so "gloves, syringes and admin charges" is caught by "admin
# charges" and not by enumerating surgical supplies.
#
# The `\b` around of/in/for is load-bearing: without it, "Bill Rs.4,50,000
# including Rs.30,000 of consumables" matches "in" inside "including" and reads
# the whole bill as non-payable.
NON_PAYABLE_PATTERNS = (
    re.compile(
        rf"{_AMOUNT}\s*\b(?:of|in|for)\b\s*[^.]{{0,60}}?"
        r"(?:consumables?|disposables?|non-?payable|admin(?:istrative)?\s*charges)",
        re.IGNORECASE,
    ),
)

# Charges the wording holds out of the proportionate deduction.
#
# `(?!rs\.|₹|inr)` stops the match reaching back past a nearer amount. In
# "Bill Rs.3,00,000 of which Rs.50,000 is pharmacy, which my policy exempts
# from proportionate deduction", a plain window binds "exempt" to the BILL and
# holds ₹3,00,000 out of the proportion instead of ₹50,000.
EXEMPT_FROM_PROPORTION_PATTERNS = (
    re.compile(
        rf"{_AMOUNT}(?:(?!rs\.|₹|inr)[^.]){{0,80}}?exempt[^.]{{0,40}}?proportion",
        re.IGNORECASE,
    ),
)

# What the room ACTUALLY cost, as opposed to what the policy allows. Without
# this the room-rent proportion rule can never fire from a question, because
# `settle()` needs both sides of the ratio.
ROOM_RENT_ACTUAL_PATTERNS = (
    re.compile(rf"(?:stayed\s+in|used|had|took)\s*(?:a|an)?\s*{_AMOUNT}[^.]{{0,15}}?room", re.IGNORECASE),
    re.compile(rf"room\s*(?:was|of|at|cost)?\s*{_AMOUNT}\s*(?:per|/|a)\s*day", re.IGNORECASE),
    re.compile(rf"{_AMOUNT}\s*(?:per|/|a)\s*day\s*room", re.IGNORECASE),
)

# The user's phrasing of the room-rent entitlement. "room limit", not "room
# rent limit", is what people actually write.
ROOM_RENT_CAP_QUESTION_PATTERNS = (
    re.compile(rf"room\s*(?:rent\s*)?limit[^.]{{0,20}}?{_AMOUNT}", re.IGNORECASE),
    re.compile(rf"against\s*(?:a|an)?\s*{_AMOUNT}[^.]{{0,20}}?limit", re.IGNORECASE),
    re.compile(rf"limit\s*of\s*{_AMOUNT}\s*(?:per|/|a)\s*day", re.IGNORECASE),
)

# The user's own terms must sit ADJACENT to their noun, not merely within a
# window. "there's a Rs.25,000 deductible and my sum insured is Rs.5,00,000"
# reads the sum insured as the deductible under any forward-looking window,
# because the deductible's amount comes before the noun, not after it.
DEDUCTIBLE_QUESTION_PATTERNS = (
    re.compile(rf"{_AMOUNT}\s*(?:voluntary\s*)?deductible", re.IGNORECASE),
    re.compile(rf"deductible\s*of\s*{_AMOUNT}", re.IGNORECASE),
)

# "capped at" is safe here in a way it is not in policy prose: a user writing it
# is describing the cap on their own claim.
SUB_LIMIT_QUESTION_PATTERNS = (
    re.compile(rf"capped\s*at\s*{_AMOUNT}", re.IGNORECASE),
    *SUB_LIMIT_PATTERNS,
)

# --- Non-health products ------------------------------------------------------

# A home policy's depreciation schedule is a BAND table, not a single figure:
#   "Up to 1 Year 10% | Up to 3 Years 20% | Up to 5 Years 30%"
# so it is read as (max_age_years, percent) rows and the narrowest band covering
# the item's age wins. A single-value pattern would grab whichever row happened
# to sit nearest the match and depreciate a 1-year-old item by 30%.
DEPRECIATION_ROW_PATTERN = re.compile(
    r"up\s*to\s*(\d+)\s*years?[^%\d]{0,20}(\d+(?:\.\d+)?)\s*%", re.IGNORECASE
)

# Item age in a question is usually written in words, not digits:
# "my three-year-old laptop", not "my 3 year old laptop".
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
ITEM_AGE_PATTERN = re.compile(
    rf"\b(\d+|{'|'.join(_WORD_NUMBERS)})[\s-]*year[\s-]*old\b", re.IGNORECASE
)

# ULIP partial-withdrawal ceiling, expressed as a share of the fund.
WITHDRAWAL_CAP_PATTERNS = (
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*the\s*fund\s*value", re.IGNORECASE),
    re.compile(r"withdraw[^.]{0,60}?(\d+(?:\.\d+)?)\s*%\s*of\s*the\s*fund", re.IGNORECASE),
)

FUND_VALUE_PATTERNS = (
    re.compile(rf"fund\s*value[^.]{{0,25}}?{_AMOUNT}", re.IGNORECASE),
    re.compile(rf"fund\s*value[^.]{{0,25}}?{_LAKH}", re.IGNORECASE),
)

# What the user asked to take out, as opposed to what the fund holds. Without
# this the ULIP path reads the fund itself as the requested amount.
WITHDRAWAL_REQUEST_PATTERNS = (
    re.compile(rf"withdraw[^.]{{0,25}}?{_AMOUNT}", re.IGNORECASE),
    re.compile(rf"take\s*out[^.]{{0,25}}?{_AMOUNT}", re.IGNORECASE),
)

# Broader than the document version: a user writes "policy is 18 months old",
# "policy has now run 48 months" and "policy 48 months old" interchangeably.
POLICY_AGE_PATTERNS = (
    re.compile(r"policy[^.]{0,25}?(\d+)\s*months", re.IGNORECASE),
    re.compile(r"(\d+)\s*months[^.]{0,15}?(?:old|in force|of cover)", re.IGNORECASE),
)


def parse_amount(text: str) -> float | None:
    """Parse one Indian-format rupee amount from a fragment of text.

    Handles `Rs.2,40,000`, `₹2,40,000`, `2.4 lakh`, `1 crore` and bare digits
    with Indian grouping. Returns None rather than guessing, so a caller can
    tell "no amount here" from "zero rupees".

    Args:
        text: A short fragment expected to contain one amount.

    Returns:
        The amount in rupees, or None when nothing parseable was found.
    """
    lowered = text.lower()

    match = re.search(_CRORE, lowered)
    if match:
        return float(match.group(1)) * 10_000_000

    match = re.search(_LAKH, lowered)
    if match:
        return float(match.group(1)) * 100_000

    match = re.search(_AMOUNT, lowered)
    if match:
        return float(match.group(1).replace(",", ""))

    # Bare Indian-grouped number, e.g. "a bill of 2,40,000". Requires a comma so
    # a bare "63" (an age) or "36" (a month count) can never be read as rupees.
    match = re.search(r"\b(\d{1,2}(?:,\d{2})+,\d{3})\b", lowered)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    """Return the first capture group matched by any pattern, or None.

    Args:
        patterns: Patterns tried in order.
        text: Text to search.

    Returns:
        The captured string, or None when no pattern matched.
    """
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _first_amount(patterns: tuple[re.Pattern[str], ...], text: str) -> float | None:
    """Return the rupee amount inside the first matching span.

    Re-parses the whole matched span with `parse_amount` rather than trusting a
    capture group, so one pattern tuple can mix `Rs.3,00,000` with `2.4 lakh`
    without the caller having to know which group meant which unit.

    Args:
        patterns: Patterns tried in order. Each must span its own amount.
        text: Text to search.

    Returns:
        The amount in rupees, or None when nothing matched or parsed.
    """
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            amount = parse_amount(match.group(0))
            if amount:
                return amount
    return None


def extract_terms(chunks: list[Any]) -> dict[str, PolicyTerm]:
    """Read policy terms out of retrieved chunks, keeping each one's page.

    Scans chunks in rank order and keeps the **first** value found for each term,
    so the highest-ranked passage wins. Later chunks cannot overwrite an earlier
    reading — a policy states its co-payment once, and a second match usually
    means a different product variant in the same document.

    Args:
        chunks: Retrieved chunks, best first. Read via `getattr` so this works on
            `RetrievedChunk` without importing `phase1_rag`.

    Returns:
        Term name to `PolicyTerm`. Missing terms are simply absent, never zero —
        `claims_calculator` distinguishes "not read" from "not applicable" and
        that distinction has to survive this far.
    """
    found: dict[str, PolicyTerm] = {}

    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        page = int(getattr(chunk, "page", 0) or 0)
        if not text or page < 1:
            continue

        if "co_pay_percent" not in found:
            raw = _first_match(CO_PAY_PATTERNS, text)
            if raw and 0 < float(raw) <= 100:
                found["co_pay_percent"] = PolicyTerm(
                    value=float(raw), source_page=page, label=f"Co-payment {raw}%"
                )

        if "waiting_period_months" not in found:
            raw = _first_match(PED_WAITING_PATTERNS, text)
            # Waiting periods in Indian health wordings run 12-48 months. A
            # bound is cheap and rejects stray numbers the window let through.
            if raw and 1 <= int(raw) <= 60:
                found["waiting_period_months"] = PolicyTerm(
                    value=float(raw),
                    source_page=page,
                    label=f"Pre-existing disease waiting period {raw} months",
                )

        if "room_rent_cap_per_day" not in found:
            raw = _first_match(ROOM_RENT_CAP_PATTERNS, text)
            if raw:
                amount = parse_amount(f"rs {raw}")
                if amount:
                    found["room_rent_cap_per_day"] = PolicyTerm(
                        value=amount, source_page=page, label=f"Room rent cap ₹{amount:,.0f}/day"
                    )

        if "sum_insured" not in found:
            for pattern in SUM_INSURED_PATTERNS:
                match = pattern.search(text)
                if match:
                    amount = parse_amount(match.group(0))
                    if amount:
                        found["sum_insured"] = PolicyTerm(
                            value=amount, source_page=page, label=f"Sum insured ₹{amount:,.0f}"
                        )
                    break

        if "deductible" not in found:
            raw = _first_match(DEDUCTIBLE_PATTERNS, text)
            if raw:
                amount = parse_amount(f"rs {raw}")
                if amount:
                    found["deductible"] = PolicyTerm(
                        value=amount, source_page=page, label=f"Deductible ₹{amount:,.0f}"
                    )

        if "sub_limit" not in found:
            amount = _first_amount(SUB_LIMIT_PATTERNS, text)
            if amount:
                found["sub_limit"] = PolicyTerm(
                    value=amount, source_page=page, label=f"Sub-limit ₹{amount:,.0f}"
                )

        if "withdrawal_cap_percent" not in found:
            raw = _first_match(WITHDRAWAL_CAP_PATTERNS, text)
            if raw and 0 < float(raw) <= 100:
                found["withdrawal_cap_percent"] = PolicyTerm(
                    value=float(raw),
                    source_page=page,
                    label=f"Partial withdrawal limit {raw}% of fund value",
                )

    LOGGER.info("Extracted %d policy term(s) from %d chunk(s): %s", len(found), len(chunks), sorted(found))
    return found


def parse_item_age_years(question: str) -> int | None:
    """Read how old the claimed item is, in years.

    Args:
        question: The user's question.

    Returns:
        Age in years, or None when the question does not say.
    """
    match = ITEM_AGE_PATTERN.search(question)
    if not match:
        return None
    raw = match.group(1).lower()
    return _WORD_NUMBERS.get(raw, int(raw) if raw.isdigit() else None)


def depreciation_for_age(chunks: list[Any], age_years: int) -> PolicyTerm | None:
    """Look up the depreciation percentage covering an item of this age.

    Reads every `Up to N Years — P%` row it can find and returns the **narrowest
    band that still covers the item**, which is how a depreciation schedule is
    meant to be read. Picking the nearest row in the text instead would
    depreciate a one-year-old item at the five-year rate whenever the table
    happened to be laid out that way.

    Args:
        chunks: Retrieved chunks, best first.
        age_years: The item's age.

    Returns:
        The matching term with its page, or None when no band covers the age.
    """
    best: tuple[int, float, int] | None = None  # (max_age, percent, page)

    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        page = int(getattr(chunk, "page", 0) or 0)
        if not text or page < 1:
            continue
        for match in DEPRECIATION_ROW_PATTERN.finditer(text):
            max_age = int(match.group(1))
            percent = float(match.group(2))
            if max_age < age_years or not 0 < percent <= 100:
                continue
            if best is None or max_age < best[0]:
                best = (max_age, percent, page)

    if best is None:
        return None
    max_age, percent, page = best
    return PolicyTerm(
        value=percent,
        source_page=page,
        label=f"Depreciation for an item up to {max_age} years old",
    )


def extract_question_terms(question: str) -> dict[str, PolicyTerm]:
    """Read policy terms the user asserted in their own question.

    Users routinely supply their own terms — "my policy has a 10% co-payment
    and a Rs.50,000 deductible". Before this existed the calculator ignored
    them and echoed the bill back unchanged, which is how `t-018` settled at
    ₹300,000 against an expected ₹225,000 with both terms stated in the
    question itself and no retrieval required.

    Every term returned carries `source_page=None`. That is what stops a
    user-supplied number from being rendered as a page citation the user cannot
    check — see `PolicyTerm` in `claims_calculator`.

    Args:
        question: The user's question.

    Returns:
        Term name to `PolicyTerm`, all user-sourced. Missing terms are absent.
    """
    found: dict[str, PolicyTerm] = {}

    raw = _first_match(CO_PAY_PATTERNS, question)
    if raw and 0 < float(raw) <= 100:
        found["co_pay_percent"] = PolicyTerm(
            value=float(raw), source_page=None, label=f"Co-payment {raw}% (your figure)"
        )

    raw = _first_match(PED_WAITING_PATTERNS, question)
    if raw and 1 <= int(raw) <= 60:
        found["waiting_period_months"] = PolicyTerm(
            value=float(raw),
            source_page=None,
            label=f"Waiting period {raw} months (your figure)",
        )

    amount = _first_amount(ROOM_RENT_CAP_QUESTION_PATTERNS, question)
    if amount:
        found["room_rent_cap_per_day"] = PolicyTerm(
            value=amount, source_page=None, label=f"Room rent limit ₹{amount:,.0f}/day (your figure)"
        )

    amount = _first_amount(SUM_INSURED_PATTERNS, question)
    if amount:
        found["sum_insured"] = PolicyTerm(
            value=amount, source_page=None, label=f"Sum insured ₹{amount:,.0f} (your figure)"
        )

    amount = _first_amount(DEDUCTIBLE_QUESTION_PATTERNS, question)
    if amount:
        found["deductible"] = PolicyTerm(
            value=amount, source_page=None, label=f"Deductible ₹{amount:,.0f} (your figure)"
        )

    amount = _first_amount(SUB_LIMIT_QUESTION_PATTERNS, question)
    if amount:
        found["sub_limit"] = PolicyTerm(
            value=amount, source_page=None, label=f"Sub-limit ₹{amount:,.0f} (your figure)"
        )

    LOGGER.info("Extracted %d term(s) from the question: %s", len(found), sorted(found))
    return found


# Terms the DOCUMENT is authoritative about: policy-wide rules, stated once for
# the whole wording and identical for every holder of the policy. A user
# misremembering their co-payment must not drive the settlement.
POLICY_WIDE_TERMS = frozenset({"co_pay_percent", "waiting_period_months"})

# Everything else is plan-scoped. One PDF lists several sum-insured tiers, room
# rent bands and voluntary deductibles; retrieval cannot tell which one this user
# bought, and the user can.
#
# **Measured 2026-08-22, t-016.** The question said "sum insured Rs.5,00,000".
# Retrieval found ₹7,50,000 on p.13 — a different tier of the same product —
# and a global document-wins rule overrode the user. The ₹6,00,000 bill then sat
# under the wrong ceiling and settled at ₹600,000 instead of ₹500,000.


def merge_terms(
    question_terms: dict[str, PolicyTerm],
    document_terms: dict[str, PolicyTerm],
    policy_wide: frozenset[str] = POLICY_WIDE_TERMS,
) -> tuple[dict[str, PolicyTerm], list[str]]:
    """Combine user-asserted and document-read terms, by term-specific precedence.

    Precedence splits on what kind of number it is, not on where it came from:

    * **Policy-wide rules** (`policy_wide`) — a co-payment percentage, a waiting
      period in months. Stated once for the whole wording, identical for every
      holder. **The document wins**, because a user misremembering a rule
      produces a confidently wrong settlement.
    * **Plan-scoped amounts** — sum insured, deductible, sub-limit, room-rent
      cap. The same document lists several tiers of each and retrieval cannot
      tell which the user holds. **The user wins**, because they know which plan
      they bought and the document does not know who is asking.

    A user-supplied value stays `source_page=None` throughout, so it is never
    rendered as a citation whichever way precedence falls.

    Args:
        question_terms: Terms asserted by the user, `source_page=None`.
        document_terms: Terms read from retrieved chunks, each with a page.
        policy_wide: Term names the document is authoritative about.

    Returns:
        `(merged, conflicts)` — merged terms, and a plain-language line per
        disagreement, whichever source won.
    """
    merged = dict(question_terms)
    conflicts: list[str] = []

    for name, doc_term in document_terms.items():
        user_term = question_terms.get(name)
        disagrees = user_term is not None and abs(user_term.value - doc_term.value) > 0.001

        if user_term is None or name in policy_wide:
            merged[name] = doc_term
            if disagrees:
                conflicts.append(
                    f"You gave {name.replace('_', ' ')} as {user_term.value:,.0f}, but your "
                    f"policy states {doc_term.value:,.0f} [p.{doc_term.source_page}]. "
                    f"The policy figure was used — it is a policy-wide rule."
                )
        elif disagrees:
            # Plan-scoped: keep the user's figure, but say the document differs
            # so a genuine mismatch is visible rather than silently discarded.
            conflicts.append(
                f"You gave {name.replace('_', ' ')} as {user_term.value:,.0f}; your policy "
                f"also mentions {doc_term.value:,.0f} [p.{doc_term.source_page}], which is "
                f"likely a different plan tier. Your figure was used."
            )

    if conflicts:
        LOGGER.warning("Question/document term disagreement: %s", conflicts)
    return merged, conflicts


def extract_bill_facts(question: str) -> dict[str, float]:
    """Read the user's own numbers out of the question.

    These are facts the user supplies, not policy terms — the bill total, the
    length of stay, how long the policy has been running. They carry no page
    because they are not read from a document.

    Args:
        question: The user's question.

    Returns:
        Keyword arguments for `ClaimRequest`, omitting anything not found.
    """
    facts: dict[str, float] = {}

    # The bill must be bound to a governing noun. `parse_amount(question)` took
    # whichever rupee figure came FIRST, so "my room limit is Rs.4,000 a day ...
    # the total bill is Rs.3,00,000" was read as a ₹4,000 bill. Deliberately
    # returns nothing rather than falling back to first-or-largest: no bill
    # means the calculator does not run and the answer says so, which is the
    # under-extraction this module is designed around.
    amount = _first_amount(BILL_PATTERNS, question)
    if amount:
        facts["claimed_amount"] = amount

    # A unit-linked withdrawal has no "bill". The amount at stake is what the
    # user asked to take out, so it stands in as the claimed amount when no
    # bill noun is present. Checked second so a genuine bill always wins.
    if "claimed_amount" not in facts:
        amount = _first_amount(WITHDRAWAL_REQUEST_PATTERNS, question)
        if amount:
            facts["claimed_amount"] = amount

    amount = _first_amount(FUND_VALUE_PATTERNS, question)
    if amount:
        facts["fund_value"] = amount

    amount = _first_amount(NON_PAYABLE_PATTERNS, question)
    if amount:
        facts["non_payable_amount"] = amount

    amount = _first_amount(EXEMPT_FROM_PROPORTION_PATTERNS, question)
    if amount:
        facts["exempt_from_proportion"] = amount

    amount = _first_amount(ROOM_RENT_ACTUAL_PATTERNS, question)
    if amount:
        facts["room_rent_per_day"] = amount

    match = re.search(r"(\d+)\s*(?:nights?|days?)\s*(?:in hospital|hospitalis|stay)", question, re.IGNORECASE)
    if match:
        facts["hospitalisation_days"] = float(match.group(1))

    raw = _first_match(POLICY_AGE_PATTERNS, question)
    if raw and 0 < int(raw) <= 600:
        facts["policy_age_months"] = float(raw)

    return facts


def build_claim_request(
    question: str, chunks: list[Any]
) -> tuple[ClaimRequest | None, list[str]]:
    """Assemble a `ClaimRequest` from the question and the retrieved policy.

    Terms come from both sources, with the document taking precedence — see
    `merge_terms`. Bill facts come from the question only, because they describe
    this specific claim and cannot appear in a policy document.

    Every field of `ClaimRequest` is populated here. An earlier version passed
    five of twelve, which silently disabled the non-payable, room-rent-proportion
    and sub-limit rules no matter what the question said.

    Args:
        question: The user's question, which supplies the bill facts.
        chunks: Retrieved chunks, best first, which supply the policy terms.

    Returns:
        `(request, conflicts)`. The request is None when no bill amount could be
        read — without an amount there is nothing to settle, and inventing one
        would be exactly the failure this module is built to avoid. `conflicts`
        lists any question/document disagreements for the answer to surface.
    """
    facts = extract_bill_facts(question)
    if "claimed_amount" not in facts:
        LOGGER.info("No claim amount found in %r — calculator cannot run.", question)
        return None, []

    terms, conflicts = merge_terms(extract_question_terms(question), extract_terms(chunks))

    # Depreciation is a schedule lookup rather than a single term: the band
    # depends on how old the item is, so it cannot come out of `extract_terms`
    # with the others. Absent age or absent schedule means absent term, which
    # `settle()` treats as "not read" and reports.
    age_years = parse_item_age_years(question)
    depreciation = depreciation_for_age(chunks, age_years) if age_years is not None else None

    request = ClaimRequest(
        claimed_amount=facts["claimed_amount"],
        non_payable_amount=facts.get("non_payable_amount", 0.0),
        room_rent_per_day=facts.get("room_rent_per_day"),
        hospitalisation_days=int(facts["hospitalisation_days"])
        if "hospitalisation_days" in facts
        else None,
        policy_age_months=int(facts["policy_age_months"])
        if "policy_age_months" in facts
        else None,
        exempt_from_proportion=facts.get("exempt_from_proportion", 0.0),
        fund_value=facts.get("fund_value"),
        sum_insured=terms.get("sum_insured"),
        co_pay_percent=terms.get("co_pay_percent"),
        room_rent_cap_per_day=terms.get("room_rent_cap_per_day"),
        sub_limit=terms.get("sub_limit"),
        deductible=terms.get("deductible"),
        waiting_period_months=terms.get("waiting_period_months"),
        depreciation_percent=depreciation,
        withdrawal_cap_percent=terms.get("withdrawal_cap_percent"),
    )
    return request, conflicts


# --- Self-test ---------------------------------------------------------------


class _Chunk:
    """Minimal stand-in for a retrieved chunk, so the test needs no index."""

    def __init__(self, text: str, page: int) -> None:
        """Store the two fields extraction reads.

        Args:
            text: Chunk text.
            page: 1-indexed page.
        """
        self.text = text
        self.page = page


def _self_test() -> list[tuple[str, bool, str]]:
    """Exercise amount parsing and term extraction on real policy wording.

    The strings below are taken from the corpus, not invented, so a pattern that
    passes here has matched text the pipeline actually retrieves.

    Returns:
        One (name, passed, detail) per check.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record one comparison."""
        passed = actual == expected
        results.append((name, passed, f"expected {expected!r}, got {actual!r}"))

    # Indian digit grouping is the trap this module was written around.
    check("Rs.2,40,000 parses", parse_amount("Rs.2,40,000"), 240000.0)
    check("₹ with spaces parses", parse_amount("₹ 1,00,000"), 100000.0)
    check("lakh shorthand parses", parse_amount("5 lakh"), 500000.0)
    check("decimal lakh parses", parse_amount("2.4 lakh"), 240000.0)
    check("crore parses", parse_amount("1 crore"), 10000000.0)
    check("bare Indian grouping parses", parse_amount("a bill of 2,40,000"), 240000.0)
    check("no amount returns None", parse_amount("I am 63 years old"), None)
    check("bare age is not an amount", parse_amount("aged 61 and above"), None)

    # Real wording from starhealth__health__comprehensive.pdf p.39.
    co_pay_text = (
        "I. Co-payment: This policy is subject to co-payment of 10% of each and "
        "every claim amount for fresh as well as renewal policies for Insured "
        "Persons whose age at the time of entry is 61 years and above."
    )
    terms = extract_terms([_Chunk(co_pay_text, 39)])
    check("co-payment extracted", terms["co_pay_percent"].value, 10.0)
    check("co-payment carries its page", terms["co_pay_percent"].source_page, 39)

    # Real wording from the same document, p.31.
    ped_text = (
        "Coverage under the policy after the expiry of 36 months for any pre-existing "
        "disease is subject to the same being declared at the time of application."
    )
    terms = extract_terms([_Chunk(ped_text, 31)])
    check("PED waiting period extracted", terms["waiting_period_months"].value, 36.0)
    check("waiting period carries its page", terms["waiting_period_months"].source_page, 31)

    # Rank order must decide: the first chunk's value wins.
    both = extract_terms([_Chunk(co_pay_text, 39), _Chunk("co-payment of 20%", 41)])
    check("first match wins over later ones", both["co_pay_percent"].value, 10.0)

    # Under-extraction is the designed behaviour, not a bug.
    check("unrelated text yields no terms", extract_terms([_Chunk("The Company shall pay.", 5)]), {})
    check(
        "a number without its governing noun is ignored",
        extract_terms([_Chunk("The waiting period is described in Section 3.", 7)]),
        {},
    )

    # A term with no usable page must never be fabricated into one.
    check("page 0 chunk is skipped", extract_terms([_Chunk(co_pay_text, 0)]), {})

    facts = extract_bill_facts("I'm 63 and my hospital bill is Rs.2,40,000. What do I get back?")
    check("bill amount read from question", facts.get("claimed_amount"), 240000.0)
    check("age is not read as an amount", "policy_age_months" in facts, False)

    request, _ = build_claim_request(
        "My hospital bill is Rs.2,40,000, what will I get back?", [_Chunk(co_pay_text, 39)]
    )
    check("request built", request is not None, True)
    if request:
        check("request carries the bill", request.claimed_amount, 240000.0)
        check("request carries the co-payment", request.co_pay_percent.value, 10.0)

    check(
        "no amount means no request",
        build_claim_request("Is knee surgery covered?", [_Chunk(co_pay_text, 39)])[0],
        None,
    )

    # --- The bill is bound to its noun, not to position --------------------
    #
    # Each of these is the exact wording of a task the pipeline got wrong. The
    # old `parse_amount(question)` returned the FIRST rupee figure, which is a
    # policy term in three of the four.

    check(
        "a leading room limit is not the bill",
        extract_bill_facts(
            "My SBI Alpha room limit is Rs.4,000 a day but I stayed in a Rs.6,000 room. "
            "The total bill is Rs.3,00,000."
        ).get("claimed_amount"),
        300000.0,
    )
    check(
        "a leading sub-limit is not the bill",
        extract_bill_facts(
            "I had a modern treatment procedure capped at Rs.1,25,000 under my Star policy. "
            "The bill was Rs.3,00,000 and I'm 63."
        ).get("claimed_amount"),
        300000.0,
    )
    check(
        "a leading sum insured is not the bill",
        extract_bill_facts(
            "Star policy, I'm 65, sum insured Rs.5,00,000, and the bill came to Rs.6,00,000."
        ).get("claimed_amount"),
        600000.0,
    )
    check(
        "no governing noun means no bill",
        extract_bill_facts("My policy has a 10% co-payment and a Rs.50,000 deductible.").get(
            "claimed_amount"
        ),
        None,
    )

    # --- Bill facts that were never extracted at all -----------------------

    facts = extract_bill_facts(
        "Bill Rs.2,40,000 and the hospital listed Rs.18,000 of gloves, syringes and admin charges."
    )
    check("non-payables read from the question", facts.get("non_payable_amount"), 18000.0)
    check("the bill survives alongside them", facts.get("claimed_amount"), 240000.0)

    facts = extract_bill_facts(
        "Bill Rs.3,00,000 of which Rs.50,000 is pharmacy, which my policy exempts from "
        "proportionate deduction. Room was Rs.6,000/day against a Rs.4,000 limit."
    )
    check("proportion exemption read", facts.get("exempt_from_proportion"), 50000.0)
    check("actual room rent read", facts.get("room_rent_per_day"), 6000.0)

    check(
        "room rent read from 'stayed in'",
        extract_bill_facts("I stayed in a Rs.6,000 room.").get("room_rent_per_day"),
        6000.0,
    )
    check(
        "room rent read from 'used a .../day room'",
        extract_bill_facts("I used a Rs.5,000/day room against a Rs.4,000 limit.").get(
            "room_rent_per_day"
        ),
        5000.0,
    )

    # Three phrasings of policy age, all of which the old single pattern missed
    # except the first.
    check(
        "policy age: 'policy is N months old'",
        extract_bill_facts("My Star policy is 18 months old.").get("policy_age_months"),
        18.0,
    )
    check(
        "policy age: 'policy has now run N months'",
        extract_bill_facts("My Star policy has now run 48 months.").get("policy_age_months"),
        48.0,
    )
    check(
        "policy age: 'policy N months old'",
        extract_bill_facts("Star policy, I'm 63, policy 48 months old.").get("policy_age_months"),
        48.0,
    )

    # --- Terms the user asserts in the question ----------------------------

    q_terms = extract_question_terms(
        "Bill of Rs.3,00,000, my policy has a Rs.50,000 deductible and a 10% co-payment."
    )
    check("user co-payment extracted", q_terms["co_pay_percent"].value, 10.0)
    check("user deductible extracted", q_terms["deductible"].value, 50000.0)
    check("user term carries no page", q_terms["co_pay_percent"].source_page, None)
    check("user term knows it is user-sourced", q_terms["co_pay_percent"].from_user, True)

    q_terms = extract_question_terms("the procedure is capped at Rs.3,00,000")
    check("user sub-limit extracted", q_terms["sub_limit"].value, 300000.0)

    q_terms = extract_question_terms("I used a Rs.5,000/day room against a Rs.4,000 limit")
    check("user room-rent limit extracted", q_terms["room_rent_cap_per_day"].value, 4000.0)

    check(
        "a question with no terms yields none",
        extract_question_terms("Is my knee surgery covered?"),
        {},
    )

    # --- Precedence: the document wins -------------------------------------

    merged, conflicts = merge_terms(
        extract_question_terms("my policy has a 20% co-payment"),
        extract_terms([_Chunk(co_pay_text, 39)]),
    )
    check("document value overrides the user's", merged["co_pay_percent"].value, 10.0)
    check("overridden term is citable again", merged["co_pay_percent"].source_page, 39)
    check("the disagreement is reported", len(conflicts), 1)

    merged, conflicts = merge_terms(
        extract_question_terms("my policy has a 10% co-payment"),
        extract_terms([_Chunk(co_pay_text, 39)]),
    )
    check("agreement raises no conflict", conflicts, [])

    merged, conflicts = merge_terms(
        extract_question_terms("my policy has a 20% co-payment"), {}
    )
    check("an unretrieved term keeps the user's value", merged["co_pay_percent"].value, 20.0)
    check("and stays uncitable", merged["co_pay_percent"].source_page, None)
    check("and is not a conflict", conflicts, [])

    # --- Plan-scoped amounts go the other way ------------------------------
    #
    # This is t-016. The question states a ₹5,00,000 sum insured; retrieval finds
    # a ₹7,50,000 tier of the same product on p.13. Under a global document-wins
    # rule the ₹6,00,000 bill sat under the wrong ceiling and settled at
    # ₹600,000 instead of ₹500,000.
    doc_si = {"sum_insured": PolicyTerm(750_000.0, 13, "Sum insured")}
    merged, conflicts = merge_terms(
        extract_question_terms("Star policy, I'm 65, sum insured Rs.5,00,000"), doc_si
    )
    check("a plan-scoped amount keeps the user's tier", merged["sum_insured"].value, 500000.0)
    check("and stays uncitable", merged["sum_insured"].source_page, None)
    check("but the other tier is still reported", len(conflicts), 1)

    # The same input under the old rule would have produced 750000. Guard the
    # direction explicitly so a future refactor cannot silently flip it back.
    check(
        "the document tier did not win",
        merged["sum_insured"].value != 750000.0,
        True,
    )

    # A plan-scoped term the user did NOT state still comes from the document,
    # because something is better than nothing and it is citable.
    merged, conflicts = merge_terms({}, doc_si)
    check("an unstated plan term falls back to the document", merged["sum_insured"].value, 750000.0)
    check("and is citable", merged["sum_insured"].source_page, 13)
    check("and is not a conflict", conflicts, [])

    # Agreement across the split raises nothing either way.
    merged, conflicts = merge_terms(
        extract_question_terms("sum insured Rs.7,50,000"), doc_si
    )
    check("agreeing plan terms raise no conflict", conflicts, [])

    # --- End to end: every ClaimRequest field is populated ------------------
    #
    # The old builder passed 5 of 12 fields, so non-payables, actual room rent,
    # the proportion exemption and the sub-limit were dropped whatever the
    # question said. This is t-023's wording.

    request, _ = build_claim_request(
        "Star policy, I'm 63, policy 48 months old. Bill Rs.4,50,000 including Rs.30,000 of "
        "consumables. I used a Rs.5,000/day room against a Rs.4,000 limit, the procedure is "
        "capped at Rs.3,00,000, there's a Rs.25,000 deductible and my sum insured is "
        "Rs.5,00,000. What is payable?",
        [],
    )
    check("end to end: bill", request.claimed_amount, 450000.0)
    check("end to end: non-payables", request.non_payable_amount, 30000.0)
    check("end to end: actual room rent", request.room_rent_per_day, 5000.0)
    check("end to end: policy age", request.policy_age_months, 48)
    check("end to end: room rent cap", request.room_rent_cap_per_day.value, 4000.0)
    check("end to end: sub-limit", request.sub_limit.value, 300000.0)
    check("end to end: deductible", request.deductible.value, 25000.0)
    check("end to end: sum insured", request.sum_insured.value, 500000.0)

    # --- Non-health products ------------------------------------------------
    #
    # t-027 and t-028 are marked `requires_unimplemented` in the task set: the
    # extraction and the calculator rule both had to exist for them to settle.

    check("word item age parses", parse_item_age_years("my three-year-old laptop"), 3)
    check("digit item age parses", parse_item_age_years("a 5 year old TV"), 5)
    check("no age returns None", parse_item_age_years("my laptop was stolen"), None)

    # A depreciation schedule is a band table. The narrowest covering band wins,
    # regardless of where it sits in the text.
    schedule = _Chunk(
        "Depreciation: Up to 1 Year 10%, Up to 3 Years 20%, Up to 5 Years 30%.", 5
    )
    check("narrowest covering band wins", depreciation_for_age([schedule], 3).value, 20.0)
    check("a younger item takes the lower band", depreciation_for_age([schedule], 1).value, 10.0)
    check("an older item takes the wider band", depreciation_for_age([schedule], 4).value, 30.0)
    check("beyond every band returns None", depreciation_for_age([schedule], 9), None)
    check("the band carries its page", depreciation_for_age([schedule], 3).source_page, 5)

    facts = extract_bill_facts("My three-year-old laptop worth Rs.80,000 was stolen.")
    check("'worth' reads the item value", facts.get("claimed_amount"), 80000.0)

    # t-028: no bill noun anywhere, so the withdrawal request stands in.
    facts = extract_bill_facts(
        "My ULIP fund value is Rs.8,00,000 and I want to withdraw Rs.2,00,000 this year."
    )
    check("withdrawal request becomes the claim", facts.get("claimed_amount"), 200000.0)
    check("fund value is read separately", facts.get("fund_value"), 800000.0)

    terms = extract_terms([_Chunk("Partial withdrawals are limited to 20% of the Fund Value.", 3)])
    check("withdrawal cap extracted", terms["withdrawal_cap_percent"].value, 20.0)
    check("withdrawal cap carries its page", terms["withdrawal_cap_percent"].source_page, 3)

    request, _ = build_claim_request(
        "My ULIP fund value is Rs.8,00,000 and I want to withdraw Rs.2,00,000 this year.",
        [_Chunk("Partial withdrawals are limited to 20% of the Fund Value.", 3)],
    )
    check("ULIP end to end: requested", request.claimed_amount, 200000.0)
    check("ULIP end to end: fund", request.fund_value, 800000.0)
    check("ULIP end to end: cap", request.withdrawal_cap_percent.value, 20.0)

    request, _ = build_claim_request(
        "My three-year-old laptop worth Rs.80,000 was stolen. What does the home policy pay?",
        [schedule],
    )
    check("property end to end: value", request.claimed_amount, 80000.0)
    check("property end to end: depreciation", request.depreciation_percent.value, 20.0)

    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase3_agents.term_extraction",
        description="Read calculator inputs out of retrieved policy text. Deliberately conservative.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the extraction checks. Pure functions — no index, no models, no cost.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the extraction self-test from the command line.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 on a failed check or bad usage.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.self_test:
        print("Nothing to do. Pass --self-test.")
        return 1

    results = _self_test()
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        if not passed:
            print(f"        {detail}")
    failed = sum(1 for _, passed, _ in results if not passed)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
