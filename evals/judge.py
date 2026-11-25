"""LLM judge that scores chatbot answers for accuracy, faithfulness, and completeness.

A different, stronger model than the chatbot's (gpt-4.1-mini vs gpt-4o-mini) is used to reduce
self-preference bias. Completeness is computed from per-fact yes/no checks instead of a vague score.

Faithfulness is verified in code, not taken on the judge's word: for every claim in the answer the
judge must copy the supporting text from the retrieved context, and a claim only counts as supported
if that quote is actually found in the context. (Calibration showed that a judge asked directly for
"unsupported claims" flagged correct, context-backed statements as unsupported in 6 of 8 cases.)
"""
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

JUDGE_MODEL = "gpt-4.1-mini"
MIN_QUOTE_WORDS = 3  # shorter quotes ("Yes", "300 SEK") don't prove anything
# A quote counts as found if this share of its words occurs in one retrieved passage. Exact substring
# matching failed on correct claims when the judge joined text across bullet lines or garbled å/ä/ö.
MIN_WORD_COVERAGE = 0.9
# Admissions of missing information are not factual claims (leaks are measured separately).
LACKS_INFO = re.compile(r"\b(don't|do not|doesn't|does not) have\b.*\binformation\b|\bno specific information\b", re.I)

JUDGE_SYSTEM = """You are a strict evaluator of a dental FAQ chatbot for Sweden.

You get a user question, reference facts that a complete answer should contain, the context passages the \
chatbot retrieved, and the chatbot's answer. Evaluate only the answer.

1. fact_checks: for EACH reference fact, in order, decide if the answer conveys it. Wording may differ, but the \
meaning and any numbers must match. Some facts describe behavior, such as "States that it does not have \
information specific to Umeå"; mark those covered only if the answer does that.

2. claims: split the answer into its individual factual claims. For each claim, copy into supporting_quote the \
exact words from the RETRIEVED CONTEXT that support it (copy a contiguous phrase or sentence verbatim, do not \
paraphrase). Use an empty string if nothing in the context supports it. The reference facts are NOT context: \
a claim that is not among the reference facts is fine as long as the context supports it. A claim that is a \
simple calculation from numbers in the context is supported; quote the numbers it is based on. Skip greetings, \
general phrasing, and statements that the chatbot lacks information.

3. incorrect_claims: statements in the answer that are factually wrong or contradict the context or the \
reference facts. Quote them briefly. Empty list if none.

4. correctness: 1-5. 5 = everything stated is correct; 4 = correct with a minor imprecision; 3 = partly wrong \
or misleading; 2 = mostly wrong; 1 = wrong or harmful."""

JUDGE_HUMAN = """Question:
{question}

Reference facts:
{facts}

Retrieved context:
{context}

Chatbot answer:
{answer}"""


class FactCheck(BaseModel):
    fact: str
    covered: bool


class Claim(BaseModel):
    claim: str
    supporting_quote: str


class Judgement(BaseModel):
    fact_checks: list[FactCheck]
    claims: list[Claim]
    incorrect_claims: list[str] = Field(default_factory=list)
    correctness: int = Field(ge=1, le=5)


def _normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^\w%'.,-]+", " ", text)  # drop markdown, bullets, brackets
    return re.sub(r"\s+", " ", text).strip(" .,")


def _words(text: str) -> list[str]:
    return re.findall(r"[\w%]+", _normalize(text))


def quote_in_context(quote: str, contexts: list[str]) -> bool:
    words = _words(quote)
    if len(words) < MIN_QUOTE_WORDS:
        return False
    for context in contexts:
        vocab = set(_words(context))
        if sum(w in vocab for w in words) / len(words) >= MIN_WORD_COVERAGE:
            return True
    return False


class Judge:
    def __init__(self, model: str = JUDGE_MODEL):
        llm = ChatOpenAI(model=model, temperature=0)
        prompt = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])
        self.chain = prompt | llm.with_structured_output(Judgement)

    def evaluate(self, question: str, facts: list[str], contexts: list[str], answer: str) -> tuple[Judgement, list[str]]:
        """Returns the judgement and the claims whose supporting quote was not found in the context."""
        j = self.chain.invoke({
            "question": question,
            "facts": "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1)),
            "context": "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1)) or "(none)",
            "answer": answer,
        })
        unsupported = [
            c.claim for c in j.claims
            if not LACKS_INFO.search(c.claim) and not quote_in_context(c.supporting_quote, contexts)
        ]
        return j, unsupported
