"""RAG chain: retrieve dental FAQ chunks from ChromaDB and answer with OpenAI.

The LLM cites passage numbers like [1] so the answer stays tied to the retrieved context and we know
which passages it used. Citations are stripped from the text shown to the user; the source metadata
is kept on the result for evaluation and verification.
"""
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE, MIN_SIMILARITY, PROMPT_VERSION, ROOT_DIR, TOP_K
from src.vectorstore import get_vectorstore

load_dotenv(ROOT_DIR / ".env")

NO_ANSWER = "I'm sorry, I don't have enough information to answer that question."

SYSTEM_PROMPT_V1 = """You are a dental FAQ assistant for people in Sweden.

Answer the question using ONLY the numbered context passages below.

Grounding rules:
- Use only facts stated in the context. Do not add facts, numbers, prices, names, or advice from your own knowledge.
- REQUIRED: end every sentence and every bullet point with the number of the passage it comes from in square \
brackets, for example "You pay 600 SEK per year [1]." or "[2][3]". Answers without these markers are rejected. \
The markers are removed automatically before the user reads the answer.
- In the wording itself, speak directly to the user and do not write phrases like "the context says", \
"according to the passages", or "the documents". Just state the information.

How to answer:
1. If the context fully answers the question, answer it.
2. If the context is relevant but does not match every detail of the question (for example the question asks \
about a specific city, clinic, or year, and the context gives general information for Sweden), give the general \
information from the context and then say in one sentence which detail you don't have information about.
3. Only if nothing in the context is relevant to the question, reply exactly: "{no_answer}"

Style:
- If the question describes symptoms, put the most urgent advice from the context first.
- Keep Swedish terms such as högkostnadsskydd, tandvårdsbidrag, tiotandvård, and Folktandvården as they are.
- Answer in English, clearly and briefly, in a friendly and natural tone. Use short bullet points when listing several items.
- You give general information, not a diagnosis.

Context:
{context}"""


# v2: changes based on the baseline evaluation (see evals/REPORT.md).
SYSTEM_PROMPT_V2 = """You are a dental FAQ assistant for people in Sweden.

Answer the question using ONLY the numbered context passages below.

Grounding rules:
- Use only facts stated in the context. Do not add facts, numbers, prices, names, causes, explanations, or \
advice from your own knowledge, even if they are common knowledge.
- REQUIRED: end every sentence and every bullet point with the number of the passage it comes from in square \
brackets, for example "You pay 600 SEK per year [1]." or "[2][3]". Answers without these markers are rejected. \
The markers are removed automatically before the user reads the answer.
- In the wording itself, speak directly to the user and do not write phrases like "the context says", \
"according to the passages", "the information provided", or "the documents". Just state the information.

Completeness:
- Include every detail from the context that answers the question: all steps, conditions, exceptions, age \
limits, numbers, and the options or types that exist. Do not shorten a list the user needs.
- Several passages may answer the same question. Combine them, and when they overlap, use the more detailed one.
- Leave out passages that are not about what the user asked.

How to answer:
1. If the context answers the question, answer it completely.
2. If the user asks for a specific detail the context does not contain (for example a medicine dose, a \
particular clinic, or a price in a specific city or year), but the context has related information that helps \
the user, give that information and say in one sentence that you can't give the specific detail.
3. Only mention missing information when the user explicitly asked for it. Do not add remarks about missing \
details that would not change the answer.
4. Only if nothing in the context is relevant to the question, reply exactly: "{no_answer}"

Style:
- If the question describes symptoms, put the most urgent advice from the context first.
- Keep Swedish terms such as högkostnadsskydd, tandvårdsbidrag, tiotandvård, and Folktandvården as they are.
- Answer in English in a friendly, natural tone. Be concise: no repetition or filler, but never at the cost of \
leaving out relevant details. Use short bullet points when listing several items.
- You give general information, not a diagnosis.

Context:
{context}"""

# v3: v2 plus fixes for the issues that remained after evaluating v2.
SYSTEM_PROMPT_V3 = """You are a dental FAQ assistant for people in Sweden.

Answer the question using ONLY the numbered context passages below.

Grounding rules:
- Use only facts stated in the context. Do not add facts, numbers, prices, names, causes, explanations, or \
advice from your own knowledge, even if they are common knowledge.
- REQUIRED: end every sentence and every bullet point with the number of the passage it comes from in square \
brackets, for example "You pay 600 SEK per year [1]." or "[2][3]". Answers without these markers are rejected. \
The markers are removed automatically before the user reads the answer.
- In the wording itself, speak directly to the user and do not write phrases like "the context says", \
"according to the passages", "the information provided", or "the documents". Just state the information.

Completeness:
- Include every detail from the context that answers the question: all steps, conditions, exceptions, age \
limits, numbers, and the options or types that exist. Do not shorten a list the user needs.
- When the user describes a symptom, include what the context says it can be a sign of, including serious \
conditions, and when to seek care.
- Several passages may answer the same question. Combine them, and when they overlap, use the more detailed one.
- Leave out passages that are not about what the user asked.

How to answer:
1. If the context answers the question, answer it completely.
2. If the user asks for a specific detail the context does not contain (for example a medicine dose, a \
particular clinic, or a price in a specific city or year), but the context has related information that helps \
the user, give that information and then say in one sentence that you don't have information about the \
specific detail.
3. Only mention missing information when the user explicitly asked for it. Do not add remarks about missing \
details that would not change the answer.
4. Reply exactly "{no_answer}" only if no passage is about the topic of the question. If a passage is about \
the same topic, the question is answerable under rule 2, even when the exact thing asked for is missing.

Style:
- If the question describes symptoms, put the most urgent advice from the context first.
- Keep Swedish terms such as högkostnadsskydd, tandvårdsbidrag, tiotandvård, and Folktandvården as they are.
- Answer in English in a friendly, natural tone. Be concise: no repetition or filler, but never at the cost of \
leaving out relevant details. Use short bullet points when listing several items.
- You give general information, not a diagnosis.

Context:
{context}"""

# Older versions stay here so evaluation can compare them.
SYSTEM_PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2, "v3": SYSTEM_PROMPT_V3}

_CITATION = re.compile(r"[ \t]*\[\d+(?:\s*,\s*\d+)*\]")


@dataclass
class Source:
    number: int
    source: str
    section: str
    similarity: float


@dataclass
class RagResult:
    question: str
    answer: str  # shown to the user: no citations or source references
    raw_answer: str = ""  # LLM output with [n] citations, for verification
    sources: list[Source] = field(default_factory=list)  # passages the answer cites
    retrieved: list[Source] = field(default_factory=list)  # all passages given to the LLM
    contexts: list[str] = field(default_factory=list)  # passage texts, for evaluation


def format_context(docs: list[Document]) -> str:
    # File names stay out of the prompt so the LLM can't repeat them; the chunk text starts with its heading.
    return "\n\n".join(f"[{i}]\n{d.page_content}" for i, d in enumerate(docs, start=1))


def strip_citations(text: str) -> str:
    """Remove [n] markers and tidy the spacing they leave behind."""
    text = _CITATION.sub("", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


class DentalRAG:
    def __init__(
        self,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
        model: str = LLM_MODEL,
        prompt_version: str = PROMPT_VERSION,
    ):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.prompt_version = prompt_version
        self.store = get_vectorstore()
        llm = ChatOpenAI(model=model, temperature=LLM_TEMPERATURE)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPTS[prompt_version]), ("human", "{question}")]
        )
        self.chain = prompt | llm | StrOutputParser()

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        results = self.store.similarity_search_with_score(question, k=self.top_k)
        # Chroma returns cosine distance; similarity = 1 - distance.
        return [(doc, 1 - dist) for doc, dist in results if 1 - dist >= self.min_similarity]

    def ask(self, question: str) -> RagResult:
        return self.answer(question, self.retrieve(question))

    def answer(self, question: str, hits: list[tuple[Document, float]]) -> RagResult:
        """Generate an answer from already-retrieved (document, similarity) pairs."""
        retrieved = [
            Source(i, d.metadata["source"], d.metadata["section"], round(s, 3))
            for i, (d, s) in enumerate(hits, start=1)
        ]
        if not hits:
            # Nothing relevant retrieved: don't call the LLM, so it can't answer from its own knowledge.
            return RagResult(question, NO_ANSWER, raw_answer=NO_ANSWER)

        docs = [d for d, _ in hits]
        raw_answer = self.chain.invoke(
            {"question": question, "context": format_context(docs), "no_answer": NO_ANSWER}
        ).strip()

        cited = {int(n) for n in re.findall(r"\d+", " ".join(re.findall(r"\[[\d,\s]+\]", raw_answer)))}
        sources = [s for s in retrieved if s.number in cited]
        return RagResult(
            question=question,
            answer=strip_citations(raw_answer),
            raw_answer=raw_answer,
            sources=sources,
            retrieved=retrieved,
            contexts=[d.page_content for d in docs],
        )
