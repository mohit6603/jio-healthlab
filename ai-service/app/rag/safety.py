"""Healthcare safety boundary.

A small instruction-tuned model cannot be relied on to honour "do not
diagnose". Measured on this stack: asked *"My haemoglobin is 9. Do I have
anaemia? What treatment should I take?"*, Qwen2.5-0.5B-Instruct answered
"your haemoglobin level of 9 mg/dL indicates mild anemia", recommended iron
supplements and blood transfusions, and invented the unit -- none of which was
in the retrieved context.

So the boundary is enforced structurally, in the same spirit as the
no-context path: questions asking for a diagnosis, a personal result
interpretation or treatment advice are answered with a fixed message and the
model is never called. A second, narrower check runs over generated answers to
catch leakage.

This reduces risk. It does not make the system safe for clinical use, and it is
not a substitute for a qualified clinician. Patterns are heuristics: they will
miss phrasings and can over-block. Both directions are tested.
"""

from __future__ import annotations

import re

#: Returned instead of an answer when the question crosses the boundary.
CLINICAL_BOUNDARY_MESSAGE = (
    "I can't interpret results or advise on treatment — that needs a qualified "
    "healthcare professional who knows your full clinical picture.\n\n"
    "What I can do is explain what a test measures, what the terminology "
    "means, and how the laboratory's process works. For example: \"What does a "
    "CBC measure?\" or \"What is haemoglobin?\"\n\n"
    "Please discuss your results with your doctor or the requesting clinician."
)

#: Asking the model to interpret a personal result or advise on treatment.
_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "do I have ...", "have I got ...", "am I anaemic/diabetic/..."
    re.compile(r"\bdo(?:es)?\s+(?:i|my)\b.{0,40}\bhave\b", re.I),
    re.compile(
        r"\bam\s+i\b.{0,30}\b(anaemic|anemic|diabetic|ill|sick|normal)\b", re.I
    ),
    re.compile(r"\b(diagnos\w+)\b", re.I),
    # "my <value> is ...", "my result/report/level ..."
    re.compile(
        r"\bmy\s+\w+\s+(?:level|count|value|result|reading)s?\s+(?:is|are|was)\b",
        re.I,
    ),
    re.compile(r"\bmy\s+(?:results?|reports?|numbers?|levels?)\b", re.I),
    re.compile(
        r"\bis\s+(?:my|this)\s+\w+\s+(?:normal|high|low|ok|okay|bad|serious)\b",
        re.I,
    ),
    re.compile(r"\bwhat\s+(?:do|does)\s+my\b", re.I),
    # treatment / medication advice
    re.compile(
        r"\b(?:what|which)\s+(?:treatment|medicine|medication|drug|dose)\b", re.I
    ),
    re.compile(r"\bshould\s+i\s+(?:take|stop|start|use|try)\b", re.I),
    re.compile(r"\bhow\s+(?:do|can|should)\s+i\s+(?:treat|cure|fix|lower|raise)\b", re.I),
    re.compile(r"\bwhat\s+should\s+i\s+do\s+about\b", re.I),
    re.compile(r"\b(?:prescri\w+|cure|remedy|dosage)\b", re.I),
)

#: Clinical assertions that must not appear in a generated answer. Deliberately
#: narrow -- these address the reader personally, which an informational
#: explanation never needs to do.
_ANSWER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "you have" alone is far too broad: "a sample you have already given",
    # "you have a report ready" are all innocuous. Require a condition to
    # follow, within a few words.
    re.compile(
        r"\byou\s+(?:have|likely\s+have|probably\s+have|are\s+suffering\s+from)"
        r"(?:\s+\w+){0,3}\s+"
        r"(anaemia|anemia|diabet\w+|infection|deficiency|disease|disorder|"
        r"cancer|hypothyroid\w*|hyperthyroid\w*|hypertension|"
        r"a\s+condition|an?\s+abnormality)\b",
        re.I,
    ),
    re.compile(
        r"\byour\s+\w+\s+(?:level|count|value|result)s?\b"
        r".{0,40}\b(indicat\w+|means?|suggests?|shows?)\b",
        re.I,
    ),
    re.compile(r"\byou\s+should\s+(?:take|start|stop|increase|reduce|use)\b", re.I),
    re.compile(
        r"\bi\s+(?:recommend|advise|suggest)\s+(?:you|taking|that\s+you)\b", re.I
    ),
    re.compile(
        r"\b(?:supplements?|transfusions?|medication|antibiotics?)\b"
        r".{0,30}\b(?:recommended|necessary|needed|prescribed)\b",
        re.I,
    ),
    re.compile(r"\byou\s+(?:need|require)\s+(?:treatment|medication|therapy)\b", re.I),
    re.compile(r"\bindicat\w+\s+(?:mild|moderate|severe)\b", re.I),
)


def requests_clinical_advice(question: str) -> bool:
    """Whether ``question`` asks for a diagnosis, interpretation or treatment.

    Returns True to *withhold* generation, so a false positive costs the user a
    redirect while a false negative risks unsafe output.
    """
    return any(pattern.search(question) for pattern in _REQUEST_PATTERNS)


def contains_clinical_advice(answer: str) -> bool:
    """Whether a generated answer crossed the boundary despite instructions."""
    return any(pattern.search(answer) for pattern in _ANSWER_PATTERNS)
