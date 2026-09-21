# ================================================================
# CONTRACT INTELLIGENCE SYSTEM
# ================================================================
#
# PURPOSE
# -------
# 1. Load and validate a contract PDF
# 2. Extract page-level text
# 3. Detect contract sections
# 4. Retrieve similar reference clauses from Qdrant
# 5. Perform a full contract review
# 6. Return a structured Pydantic report
# 7. Start an interactive Q&A session
# 8. User enters 0 to exit
#
# ARCHITECTURE
# ------------
#
#                    CONTRACT PDF
#                         |
#                         v
#                 PDF VALIDATION
#                         |
#                         v
#                 PDF EXTRACTION
#                         |
#                         v
#              SECTION / CLAUSE SPLIT
#                         |
#                         v
#              REFERENCE RETRIEVAL
#                  (Qdrant + Embedding)
#                         |
#                         v
#                FULL CONTRACT REVIEW
#                         |
#                         v
#                  PYDANTIC VALIDATION
#                         |
#                         v
#               STRUCTURED REPORT
#                         |
#                         v
#                  INTERACTIVE Q&A
#                         |
#              +----------+----------+
#              |                     |
#          QUESTION                  0
#              |                     |
#              v                     v
#        CONTRACT RAG             EXIT
#
# ================================================================


# ================================================================
# 1. IMPORTS
# ================================================================

import os
import re
import json
from pathlib import Path
from typing import TypedDict, List, Optional, Literal

import numpy as np

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

from langgraph.graph import StateGraph, START, END


# ================================================================
# 2. APPLICATION CONFIGURATION
# ================================================================
#
# Centralizing configuration is better than scattering constants
# throughout the application.
# ================================================================

load_dotenv()


class Settings:
    """Application configuration loaded from environment variables."""

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    GROQ_MODEL = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    QDRANT_URL = os.getenv("QDRANT_URL")

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    QDRANT_COLLECTION = os.getenv(
        "QDRANT_COLLECTION",
        "contract_clauses"
    )

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    TOP_K_REFERENCE = 5

    TOP_K_CONTRACT = 6

    TEMPERATURE = 0.1


settings = Settings()


# ================================================================
# 3. LOGGING / DISPLAY HELPERS
# ================================================================

def print_header(title: str) -> None:
    """Print a consistent section header."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_step(message: str) -> None:
    """Print a pipeline progress message."""

    print(f"\n[PIPELINE] {message}")


def print_success(message: str) -> None:
    """Print a successful operation."""

    print(f"✓ {message}")


def print_warning(message: str) -> None:
    """Print a warning."""

    print(f"⚠ {message}")


def print_error(message: str) -> None:
    """Print an error."""

    print(f"❌ {message}")


# ================================================================
# 4. ENVIRONMENT VALIDATION
# ================================================================

def validate_environment() -> None:
    """
    Validate required environment variables before starting
    the application.
    """

    required = {
        "GROQ_API_KEY": settings.GROQ_API_KEY,
        "QDRANT_URL": settings.QDRANT_URL,
        "QDRANT_API_KEY": settings.QDRANT_API_KEY,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing)
        )

    print_success("Environment variables validated")


# ================================================================
# 5. CLIENT INITIALIZATION
# ================================================================

def initialize_clients():
    """
    Initialize external services and the embedding model.
    """

    print_header("INITIALIZING CONTRACT INTELLIGENCE SYSTEM")

    validate_environment()

    # ------------------------------------------------------------
    # Groq
    # ------------------------------------------------------------

    groq_client = Groq(
        api_key=settings.GROQ_API_KEY
    )

    print_success(
        f"Groq initialized: {settings.GROQ_MODEL}"
    )

    # ------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------

    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )

    print_success("Qdrant client initialized")

    # ------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------

    print_step(
        f"Loading embedding model: "
        f"{settings.EMBEDDING_MODEL}"
    )

    embedding_model = SentenceTransformer(
        settings.EMBEDDING_MODEL
    )

    dimension = (
        embedding_model
        .get_sentence_embedding_dimension()
    )

    print_success(
        f"Embedding model loaded ({dimension} dimensions)"
    )

    return (
        groq_client,
        qdrant_client,
        embedding_model
    )


# ================================================================
# 6. QDRANT HEALTH CHECK
# ================================================================

def validate_qdrant_collection(
    qdrant_client: QdrantClient
) -> None:
    """
    Verify that the reference clause collection exists.
    """

    print_step("Checking Qdrant reference collection")

    try:

        collection = qdrant_client.get_collection(
            collection_name=settings.QDRANT_COLLECTION
        )

        print_success(
            f"Collection found: "
            f"{settings.QDRANT_COLLECTION}"
        )

        print_success(
            f"Reference clauses stored: "
            f"{collection.points_count}"
        )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to access Qdrant collection "
            f"'{settings.QDRANT_COLLECTION}': {exc}"
        )


# ================================================================
# 7. PYDANTIC INPUT SCHEMAS
# ================================================================

class ContractInput(BaseModel):
    """Validated contract input."""

    file_path: str = Field(
        min_length=1
    )


class QuestionInput(BaseModel):
    """Validated user question."""

    question: str = Field(
        min_length=3
    )


# ================================================================
# 8. PYDANTIC CONTRACT ANALYSIS SCHEMAS
# ================================================================

class ContractSection(BaseModel):
    """
    Represents one section extracted from the uploaded contract.
    """

    section_number: Optional[str] = None

    title: str

    text: str

    pages: List[int] = Field(
        default_factory=list
    )


class ReferenceClause(BaseModel):
    """
    Reference clause retrieved from Qdrant.
    """

    filename: Optional[str] = None

    clause_type: Optional[str] = None

    clause_text: str

    similarity_score: float


class RiskItem(BaseModel):
    """
    Potential contractual issue identified during review.
    """

    title: str

    severity: Literal[
        "Low",
        "Medium",
        "High",
        "Unknown"
    ]

    explanation: str

    contract_basis: str

    page: Optional[int] = None


class MissingInformation(BaseModel):
    """
    Information that appears blank, unspecified or incomplete.
    """

    item: str

    explanation: str

    page: Optional[int] = None


class ClauseAnalysis(BaseModel):
    """
    Analysis of an important contract clause.
    """

    clause_name: str

    description: str

    obligations: List[str] = Field(
        default_factory=list
    )

    potential_concerns: List[str] = Field(
        default_factory=list
    )

    page: Optional[int] = None


class ContractAnalysisReport(BaseModel):
    """
    Complete structured contract review.

    This represents a lawyer-style analytical report,
    not legal advice.
    """

    contract_type: str

    parties: List[str] = Field(
        default_factory=list
    )

    purpose: str

    executive_summary: str

    key_obligations: List[str] = Field(
        default_factory=list
    )

    important_clauses: List[ClauseAnalysis] = Field(
        default_factory=list
    )

    financial_terms: List[str] = Field(
        default_factory=list
    )

    termination_terms: List[str] = Field(
        default_factory=list
    )

    liability_and_indemnification: List[str] = Field(
        default_factory=list
    )

    insurance_requirements: List[str] = Field(
        default_factory=list
    )

    intellectual_property_and_work_product: List[str] = Field(
        default_factory=list
    )

    confidentiality_and_data_protection: List[str] = Field(
        default_factory=list
    )

    disputes_and_governing_law: List[str] = Field(
        default_factory=list
    )

    subcontracting_and_assignment: List[str] = Field(
        default_factory=list
    )

    compliance_requirements: List[str] = Field(
        default_factory=list
    )

    risks: List[RiskItem] = Field(
        default_factory=list
    )

    missing_information: List[MissingInformation] = Field(
        default_factory=list
    )

    reference_comparison: List[str] = Field(
        default_factory=list
    )

    important_pages: List[int] = Field(
        default_factory=list
    )

    review_disclaimer: str


# ================================================================
# 9. PYDANTIC Q&A OUTPUT SCHEMA
# ================================================================

class QASource(BaseModel):
    """Source used to answer a contract question."""

    source_type: Literal[
        "Uploaded Contract",
        "Reference Knowledge Base"
    ]

    title: str

    page: Optional[int] = None

    text: str


class ContractQAResponse(BaseModel):
    """Structured answer for interactive contract Q&A."""

    answer: str

    clause_type: Optional[str] = None

    risk_level: Literal[
        "Low",
        "Medium",
        "High",
        "Unknown"
    ] = "Unknown"

    explanation: str

    sources: List[QASource] = Field(
        default_factory=list
    )

    confidence: Literal[
        "High",
        "Medium",
        "Low"
    ] = "Medium"


# ================================================================
# 10. LANGGRAPH STATE
# ================================================================

class ContractState(TypedDict, total=False):

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    file_path: str

    question: str

    # ------------------------------------------------------------
    # Contract extraction
    # ------------------------------------------------------------

    page_text: List[str]

    document_text: str

    sections: List[dict]

    # ------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------

    query_vector: List[float]

    contract_results: List[dict]

    reference_results: List[dict]

    # ------------------------------------------------------------
    # LLM context
    # ------------------------------------------------------------

    analysis_context: str

    qa_context: str

    # ------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------

    report: dict

    qa_answer: dict

    # ------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------

    error: str


# ================================================================
# 11. PDF EXTRACTION SERVICE
# ================================================================

def extract_pdf(
    file_path: str
):
    """
    Extract text while preserving page boundaries.

    Page numbers are retained because contract answers should
    be traceable back to the original document.
    """

    print_step("Extracting contract PDF")

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"PDF does not exist: {file_path}"
        )

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        pages.append(text.strip())

    document_text = "\n\n".join(
        f"[PAGE {i + 1}]\n{text}"
        for i, text in enumerate(pages)
        if text.strip()
    )

    if not document_text.strip():

        raise ValueError(
            "No readable text was extracted from the PDF."
        )

    print_success(
        f"Pages extracted: {len(pages)}"
    )

    print_success(
        f"Characters extracted: {len(document_text)}"
    )

    return pages, document_text


# ================================================================
# 12. SECTION DETECTION
# ================================================================

SECTION_PATTERN = re.compile(
    r"(?m)^\s*(\d+)\.\s+([A-Z][A-Z0-9 &/\-,()]+)"
)


def detect_sections(
    page_text: List[str]
) -> List[ContractSection]:
    """
    Detect numbered contract sections.

    Example:
        1. DUTIES
        2. COMPENSATION
        3. TERM
        4. EARLY TERMINATION
    """

    print_step("Detecting contract sections")

    sections = []

    current_title = None
    current_number = None
    current_text = []
    current_pages = []

    for page_number, text in enumerate(
        page_text,
        start=1
    ):

        if not text.strip():
            continue

        matches = list(
            SECTION_PATTERN.finditer(text)
        )

        if not matches:

            if current_title:
                current_text.append(text)
                current_pages.append(page_number)

            continue

        for index, match in enumerate(matches):

            # Save previous section
            if current_title:

                sections.append(
                    ContractSection(
                        section_number=current_number,
                        title=current_title.strip(),
                        text="\n".join(current_text).strip(),
                        pages=sorted(
                            set(current_pages)
                        )
                    )
                )

            current_number = match.group(1)

            current_title = match.group(2).strip()

            start = match.end()

            if index + 1 < len(matches):

                end = matches[
                    index + 1
                ].start()

                section_text = text[
                    start:end
                ]

            else:

                section_text = text[start:]

            current_text = [
                section_text.strip()
            ]

            current_pages = [
                page_number
            ]

    # Save final section
    if current_title:

        sections.append(
            ContractSection(
                section_number=current_number,
                title=current_title,
                text="\n".join(current_text).strip(),
                pages=sorted(
                    set(current_pages)
                )
            )
        )

    # Remove tiny / invalid sections
    sections = [
        section
        for section in sections
        if len(section.text.strip()) > 50
    ]

    print_success(
        f"Detected {len(sections)} contract sections"
    )

    return sections


# ================================================================
# 13. EMBEDDING SERVICE
# ================================================================

def embed_text(
    embedding_model,
    text: str
) -> np.ndarray:
    """
    Generate normalized embedding.

    all-MiniLM-L6-v2 produces 384-dimensional vectors,
    matching the existing Qdrant collection.
    """

    vector = embedding_model.encode(
        text,
        normalize_embeddings=True
    )

    return np.asarray(
        vector,
        dtype=np.float32
    )


# ================================================================
# 14. LOCAL CONTRACT RETRIEVAL
# ================================================================

def retrieve_contract_sections(
    question: str,
    sections: List[ContractSection],
    embedding_model,
    top_k: int = 6
):
    """
    Retrieve relevant sections directly from the uploaded
    contract.

    This is separate from Qdrant because Qdrant contains the
    external/reference contract knowledge base.
    """

    if not sections:
        return []

    question_vector = embed_text(
        embedding_model,
        question
    )

    scored_sections = []

    for section in sections:

        section_vector = embed_text(
            embedding_model,
            f"{section.title}\n{section.text}"
        )

        similarity = float(
            np.dot(
                question_vector,
                section_vector
            )
        )

        scored_sections.append(
            {
                "score": similarity,
                "title": section.title,
                "text": section.text,
                "pages": section.pages,
                "section_number": section.section_number
            }
        )

    scored_sections.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_sections[:top_k]


# ================================================================
# 15. QDRANT REFERENCE RETRIEVAL
# ================================================================

def retrieve_reference_clauses(
    query: str,
    embedding_model,
    qdrant_client,
    top_k: int = 5
):
    """
    Retrieve semantically similar clauses from the offline
    reference knowledge base.
    """

    query_vector = embed_text(
        embedding_model,
        query
    ).tolist()

    results = qdrant_client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k
    ).points

    references = []

    for result in results:

        payload = result.payload or {}

        references.append(
            {
                "score": float(result.score),

                "filename": payload.get(
                    "filename"
                ),

                "clause_type": payload.get(
                    "clause_type"
                ),

                "clause_text": payload.get(
                    "clause_text",
                    ""
                )
            }
        )

    return references


# ================================================================
# 16. BUILD FULL CONTRACT REFERENCE CONTEXT
# ================================================================

def build_analysis_context(
    sections: List[ContractSection],
    embedding_model,
    qdrant_client
) -> str:
    """
    Build the context used for the initial full-contract review.

    Each important contract section is paired with semantically
    similar reference clauses.
    """

    print_step(
        "Retrieving reference clauses for contract analysis"
    )

    context_parts = []

    for section in sections:

        query = (
            f"{section.title}\n"
            f"{section.text[:5000]}"
        )

        references = retrieve_reference_clauses(
            query=query,
            embedding_model=embedding_model,
            qdrant_client=qdrant_client,
            top_k=3
        )

        context_parts.append(
            f"""
============================================================
CONTRACT SECTION
============================================================

Section:
{section.section_number or ""} {section.title}

Pages:
{section.pages}

Contract Text:
{section.text}

REFERENCE CLAUSES
------------------------------------------------------------

{
    chr(10).join(
        f'''
Reference {i + 1}
Clause Type: {ref["clause_type"]}
Filename: {ref["filename"]}
Similarity: {ref["score"]:.4f}
Text:
{ref["clause_text"]}
'''
        for i, ref in enumerate(
            references
        )
    )
}
"""
        )

    context = "\n".join(
        context_parts
    )

    print_success(
        "Contract + reference context constructed"
    )

    return context


# ================================================================
# 17. FULL CONTRACT ANALYSIS PROMPT
# ================================================================

CONTRACT_ANALYSIS_SYSTEM_PROMPT = """
You are a Contract Intelligence AI performing a structured
contract review.

Your task is to analyze the ENTIRE uploaded contract.

You are NOT a lawyer and must NOT claim to provide legal advice.

Think like a professional contract-review system:

1. Identify the contract type.
2. Identify the parties.
3. Explain the commercial purpose.
4. Summarize the agreement.
5. Identify important obligations.
6. Identify important contractual clauses.
7. Analyze payment and financial terms.
8. Analyze termination provisions.
9. Analyze liability and indemnification.
10. Analyze insurance requirements.
11. Analyze intellectual property and work-product provisions.
12. Analyze confidentiality and data protection.
13. Analyze dispute resolution and governing law.
14. Analyze subcontracting and assignment.
15. Analyze regulatory/compliance requirements.
16. Identify potential risk areas.
17. Identify blank, missing or unspecified information.
18. Compare the contract conceptually with retrieved reference
    clauses.

IMPORTANT EVIDENCE RULES:

- The uploaded contract is the PRIMARY source.
- Retrieved reference clauses are SECONDARY sources.
- Never invent a clause.
- Never claim a provision exists if it is not present.
- Similarity between two clauses does NOT prove identical legal
  meaning.
- If information is absent, explicitly say it is absent or
  unspecified.
- Preserve important contractual terminology.
- Cite page numbers whenever the source information supports it.
- Do not manufacture page numbers.
- Risk levels must be based on actual contractual language.
- Do not provide unsupported legal conclusions.

The final answer must follow the supplied JSON schema.
"""


# ================================================================
# 18. FULL CONTRACT ANALYSIS NODE
# ================================================================

def analyze_contract_node(
    state: ContractState,
    groq_client: Groq
) -> ContractState:
    """
    Generate the complete structured contract review.
    """

    print_step(
        "Running full contract analysis with Groq"
    )

    context = state["analysis_context"]

    user_prompt = f"""
Analyze the following uploaded contract.

============================================================
UPLOADED CONTRACT + REFERENCE CONTEXT
============================================================

{context}

============================================================
TASK
============================================================

Produce a complete structured contract review.

The report should read like a professional contract-review
summary.

Do not give generic legal advice.

Every finding must be grounded in the supplied contract.

Pay particular attention to:

- parties
- purpose
- obligations
- payment
- term
- termination
- indemnification
- insurance
- work products
- intellectual property
- confidentiality
- disputes
- governing law
- subcontracting
- assignment
- compliance
- missing information
- potential risk areas
"""

    try:

        completion = groq_client.chat.completions.create(

            model=settings.GROQ_MODEL,

            temperature=settings.TEMPERATURE,

            messages=[
                {
                    "role": "system",
                    "content":
                        CONTRACT_ANALYSIS_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            response_format={
                "type": "json_schema",

                "json_schema": {
                    "name":
                        "contract_analysis_report",

                    "strict": False,

                    "schema":
                        ContractAnalysisReport
                        .model_json_schema()
                }
            }
        )

        raw_output = (
            completion
            .choices[0]
            .message
            .content
        )

        parsed_output = json.loads(
            raw_output
        )

        validated_report = (
            ContractAnalysisReport
            .model_validate(
                parsed_output
            )
        )

        print_success(
            "Contract analysis generated"
        )

        print_success(
            "Pydantic report validation passed"
        )

        return {
            **state,
            "report":
                validated_report.model_dump()
        }

    except Exception as exc:

        print_error(
            f"Contract analysis failed: {exc}"
        )

        return {
            **state,
            "error":
                str(exc)
        }


# ================================================================
# 19. Q&A QUERY REWRITE
# ================================================================

def rewrite_question(
    question: str,
    groq_client: Groq
) -> str:
    """
    Convert a natural-language question into a retrieval-oriented
    query.
    """

    system_prompt = """
You are a contract retrieval query optimizer.

Rewrite the user's question into a precise search query.

Focus on:

- clause type
- obligations
- rights
- restrictions
- payment
- termination
- liability
- indemnification
- insurance
- intellectual property
- confidentiality
- disputes
- governing law
- assignment
- subcontracting
- compliance

Do not answer the question.

Return ONLY the search query.
"""

    response = groq_client.chat.completions.create(

        model=settings.GROQ_MODEL,

        temperature=0.1,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ================================================================
# 20. Q&A NODE
# ================================================================

def answer_question(
    question: str,
    sections: List[ContractSection],
    embedding_model,
    qdrant_client,
    groq_client
) -> ContractQAResponse:
    """
    Answer a question using BOTH:

    1. Uploaded contract
    2. Reference clause knowledge base
    """

    validated_question = QuestionInput(
        question=question
    )

    rewritten_query = rewrite_question(
        validated_question.question,
        groq_client
    )

    # ------------------------------------------------------------
    # Retrieve relevant uploaded-contract sections
    # ------------------------------------------------------------

    contract_results = retrieve_contract_sections(

        question=rewritten_query,

        sections=sections,

        embedding_model=embedding_model,

        top_k=settings.TOP_K_CONTRACT
    )

    # ------------------------------------------------------------
    # Retrieve reference clauses
    # ------------------------------------------------------------

    reference_results = retrieve_reference_clauses(

        query=rewritten_query,

        embedding_model=embedding_model,

        qdrant_client=qdrant_client,

        top_k=settings.TOP_K_REFERENCE
    )

    # ------------------------------------------------------------
    # Build Q&A context
    # ------------------------------------------------------------

    contract_context = "\n\n".join(

        f"""
CONTRACT SECTION
Title: {item["title"]}
Pages: {item["pages"]}
Similarity: {item["score"]:.4f}

Text:
{item["text"]}
"""
        for item in contract_results
    )

    reference_context = "\n\n".join(

        f"""
REFERENCE CLAUSE
Clause Type: {item["clause_type"]}
Filename: {item["filename"]}
Similarity: {item["score"]:.4f}

Text:
{item["clause_text"]}
"""
        for item in reference_results
    )

    # ------------------------------------------------------------
    # LLM prompt
    # ------------------------------------------------------------

    system_prompt = """
You are a Contract Intelligence AI answering questions about
an uploaded contract.

The uploaded contract is the PRIMARY source.

Reference clauses are SECONDARY supporting material.

Rules:

1. Answer only from the supplied evidence.
2. Do not invent contractual provisions.
3. If the answer is not present, say so.
4. Clearly distinguish contract evidence from reference material.
5. Do not assume similar clauses have identical legal meaning.
6. Mention the relevant section when possible.
7. Include a page number when supported by the evidence.
8. Risk level must be based on the contract evidence.
9. Do not provide unsupported legal conclusions.
10. This is contract analysis, not legal advice.

Return the exact structured schema.
"""

    user_prompt = f"""
USER QUESTION
============================================================
{validated_question.question}


REWRITTEN RETRIEVAL QUERY
============================================================
{rewritten_query}


UPLOADED CONTRACT
============================================================
{contract_context}


REFERENCE KNOWLEDGE BASE
============================================================
{reference_context}


TASK
============================================================
Answer the user's question using the uploaded contract as
the primary source.

Use reference clauses only as supporting comparison material.
"""

    completion = groq_client.chat.completions.create(

        model=settings.GROQ_MODEL,

        temperature=settings.TEMPERATURE,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        response_format={
            "type": "json_schema",

            "json_schema": {
                "name":
                    "contract_qa_response",

                "strict": False,

                "schema":
                    ContractQAResponse
                    .model_json_schema()
            }
        }
    )

    raw_output = (
        completion
        .choices[0]
        .message
        .content
    )

    parsed_output = json.loads(
        raw_output
    )

    return ContractQAResponse.model_validate(
        parsed_output
    )


# ================================================================
# 21. REPORT DISPLAY
# ================================================================

def display_report(
    report: dict
) -> None:
    """
    Convert the structured Pydantic report into a human-readable
    contract-review report.
    """

    print_header(
        "CONTRACT ANALYSIS REPORT"
    )

    print(
        f"\nCONTRACT TYPE\n"
        f"{report['contract_type']}"
    )

    print(
        "\nPARTIES"
    )

    for party in report["parties"]:
        print(f"  • {party}")

    print(
        "\nPURPOSE\n"
        f"{report['purpose']}"
    )

    print(
        "\nEXECUTIVE SUMMARY\n"
        f"{report['executive_summary']}"
    )

    # ------------------------------------------------------------
    # Key obligations
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print("KEY OBLIGATIONS")

    for obligation in report[
        "key_obligations"
    ]:

        print(
            f"  • {obligation}"
        )

    # ------------------------------------------------------------
    # Important clauses
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print("IMPORTANT CLAUSES")

    for clause in report[
        "important_clauses"
    ]:

        page = (
            f" | Page {clause['page']}"
            if clause["page"]
            else ""
        )

        print(
            f"\n[{clause['clause_name']}]"
            f"{page}"
        )

        print(
            f"  {clause['description']}"
        )

        if clause["obligations"]:

            print("  Obligations:")

            for item in clause[
                "obligations"
            ]:

                print(
                    f"    • {item}"
                )

        if clause["potential_concerns"]:

            print("  Concerns:")

            for item in clause[
                "potential_concerns"
            ]:

                print(
                    f"    • {item}"
                )

    # ------------------------------------------------------------
    # Financial
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print("FINANCIAL TERMS")

    for item in report[
        "financial_terms"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print("TERMINATION")

    for item in report[
        "termination_terms"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # Liability
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "LIABILITY & INDEMNIFICATION"
    )

    for item in report[
        "liability_and_indemnification"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # Insurance
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print("INSURANCE REQUIREMENTS")

    for item in report[
        "insurance_requirements"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # IP
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "INTELLECTUAL PROPERTY & WORK PRODUCT"
    )

    for item in report[
        "intellectual_property_and_work_product"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # Confidentiality
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "CONFIDENTIALITY & DATA PROTECTION"
    )

    for item in report[
        "confidentiality_and_data_protection"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # Disputes
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "DISPUTES & GOVERNING LAW"
    )

    for item in report[
        "disputes_and_governing_law"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # Subcontracting
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "SUBCONTRACTING & ASSIGNMENT"
    )

    for item in report[
        "subcontracting_and_assignment"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print("COMPLIANCE REQUIREMENTS")

    for item in report[
        "compliance_requirements"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # Risks
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "POTENTIAL RISK AREAS"
    )

    for risk in report[
        "risks"
    ]:

        page = (
            f" | Page {risk['page']}"
            if risk["page"]
            else ""
        )

        print(
            f"\n[{risk['severity']}] "
            f"{risk['title']}{page}"
        )

        print(
            f"  {risk['explanation']}"
        )

        print(
            f"  Contract basis: "
            f"{risk['contract_basis']}"
        )

    # ------------------------------------------------------------
    # Missing information
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "MISSING / UNSPECIFIED INFORMATION"
    )

    if not report[
        "missing_information"
    ]:

        print(
            "  No obvious missing information identified."
        )

    else:

        for item in report[
            "missing_information"
        ]:

            print(
                f"\n  • {item['item']}"
            )

            print(
                f"    {item['explanation']}"
            )

    # ------------------------------------------------------------
    # Reference comparison
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "REFERENCE KNOWLEDGE BASE COMPARISON"
    )

    for item in report[
        "reference_comparison"
    ]:

        print(
            f"  • {item}"
        )

    # ------------------------------------------------------------
    # Important pages
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "IMPORTANT PAGES"
    )

    print(
        "  "
        + ", ".join(
            str(page)
            for page in report[
                "important_pages"
            ]
        )
    )

    # ------------------------------------------------------------
    # Disclaimer
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "NOTE"
    )

    print(
        f"  {report['review_disclaimer']}"
    )


# ================================================================
# 22. Q&A DISPLAY
# ================================================================

def display_qa_answer(
    response: ContractQAResponse
) -> None:
    """
    Display a structured Q&A response.
    """

    print(
        "\n" + "-" * 80
    )

    print("ANSWER")

    print(
        f"\n{response.answer}"
    )

    print(
        f"\nCLAUSE TYPE: "
        f"{response.clause_type or 'Not specified'}"
    )

    print(
        f"RISK LEVEL: "
        f"{response.risk_level}"
    )

    print(
        "\nEXPLANATION"
    )

    print(
        response.explanation
    )

    print(
        "\nSOURCES"
    )

    for source in response.sources:

        page = (
            f"Page {source.page}"
            if source.page
            else "Page not specified"
        )

        print(
            f"\n[{source.source_type}] "
            f"{source.title} | {page}"
        )

        print(
            f"  {source.text[:500]}"
        )

    print(
        f"\nCONFIDENCE: "
        f"{response.confidence}"
    )


# ================================================================
# 23. INITIAL CONTRACT ANALYSIS LANGGRAPH NODE
# ================================================================

def extract_contract_node(
    state: ContractState
) -> ContractState:
    """
    LangGraph node responsible for PDF extraction.
    """

    try:

        pages, document_text = extract_pdf(
            state["file_path"]
        )

        return {
            **state,
            "page_text": pages,
            "document_text": document_text
        }

    except Exception as exc:

        return {
            **state,
            "error": str(exc)
        }


# ================================================================
# 24. SECTION DETECTION NODE
# ================================================================

def section_detection_node(
    state: ContractState
) -> ContractState:
    """
    LangGraph node responsible for section extraction.
    """

    try:

        sections = detect_sections(
            state["page_text"]
        )

        if not sections:

            raise ValueError(
                "No meaningful contract sections "
                "could be detected."
            )

        return {
            **state,
            "sections": [
                section.model_dump()
                for section in sections
            ]
        }

    except Exception as exc:

        return {
            **state,
            "error": str(exc)
        }


# ================================================================
# 25. REFERENCE RETRIEVAL NODE
# ================================================================

def reference_retrieval_node(
    state: ContractState,
    embedding_model,
    qdrant_client
) -> ContractState:
    """
    LangGraph node that retrieves reference clauses for the
    complete contract.
    """

    try:

        sections = [
            ContractSection.model_validate(
                section
            )
            for section in state[
                "sections"
            ]
        ]

        context = build_analysis_context(

            sections=sections,

            embedding_model=embedding_model,

            qdrant_client=qdrant_client
        )

        return {
            **state,
            "analysis_context": context
        }

    except Exception as exc:

        return {
            **state,
            "error": str(exc)
        }


# ================================================================
# 26. ERROR ROUTING
# ================================================================

def route_after_node(
    state: ContractState
):
    """
    Route to END if any node has produced an error.
    """

    if state.get("error"):

        return "error"

    return "continue"


def error_node(
    state: ContractState
) -> ContractState:
    """
    Centralized error display node.
    """

    print_error(
        state.get(
            "error",
            "Unknown pipeline error"
        )
    )

    return state


# ================================================================
# 27. BUILD INITIAL ANALYSIS GRAPH
# ================================================================

def build_analysis_graph(
    groq_client,
    embedding_model,
    qdrant_client
):
    """
    Build the LangGraph responsible for the initial
    contract analysis.
    """

    graph = StateGraph(
        ContractState
    )

    # ------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------

    graph.add_node(
        "extract_contract",
        extract_contract_node
    )

    graph.add_node(
        "detect_sections",
        section_detection_node
    )

    graph.add_node(
        "retrieve_references",
        lambda state:
            reference_retrieval_node(
                state,
                embedding_model,
                qdrant_client
            )
    )

    graph.add_node(
        "analyze_contract",
        lambda state:
            analyze_contract_node(
                state,
                groq_client
            )
    )

    graph.add_node(
        "error",
        error_node
    )

    # ------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------

    graph.add_edge(
        START,
        "extract_contract"
    )

    graph.add_conditional_edges(
        "extract_contract",
        route_after_node,
        {
            "continue":
                "detect_sections",

            "error":
                "error"
        }
    )

    graph.add_conditional_edges(
        "detect_sections",
        route_after_node,
        {
            "continue":
                "retrieve_references",

            "error":
                "error"
        }
    )

    graph.add_conditional_edges(
        "retrieve_references",
        route_after_node,
        {
            "continue":
                "analyze_contract",

            "error":
                "error"
        }
    )

    graph.add_conditional_edges(
        "analyze_contract",
        route_after_node,
        {
            "continue":
                END,

            "error":
                "error"
        }
    )

    graph.add_edge(
        "error",
        END
    )

    return graph.compile()


# ================================================================
# 28. RUN INITIAL CONTRACT ANALYSIS
# ================================================================

def run_contract_analysis(
    app,
    pdf_path: str
):
    """
    Execute the full contract-analysis pipeline.
    """

    validated_input = ContractInput(
        file_path=pdf_path
    )

    print_header(
        "STARTING FULL CONTRACT REVIEW"
    )

    initial_state: ContractState = {

        "file_path":
            validated_input.file_path
    }

    final_state = app.invoke(
        initial_state
    )

    if final_state.get("error"):

        print_error(
            final_state["error"]
        )

        return None, None

    report = final_state.get(
        "report"
    )

    sections = [
        ContractSection.model_validate(
            section
        )
        for section in final_state[
            "sections"
        ]
    ]

    if not report:

        print_error(
            "No contract report was generated."
        )

        return None, None

    return report, sections


# ================================================================
# 29. INTERACTIVE Q&A LOOP
# ================================================================

def start_qa_loop(
    sections,
    groq_client,
    embedding_model,
    qdrant_client
):
    """
    Keep accepting questions until the user enters 0.
    """

    print_header(
        "CONTRACT Q&A READY"
    )

    print(
        """
You can now ask questions about the uploaded contract.

Examples:

• What are the termination conditions?
• What insurance is required?
• Who owns the work produced under this agreement?
• What are the payment obligations?
• What happens if the consultant breaches the contract?
• What are the major contractual risks?
• What is the governing law?

Enter 0 to exit.
"""
    )

    while True:

        try:

            question = input(
                "\nAsk your question: "
            ).strip()

            # ----------------------------------------------------
            # EXIT CONDITION
            # ----------------------------------------------------

            if question == "0":

                print(
                    "\n" + "=" * 80
                )

                print(
                    "EXITING CONTRACT INTELLIGENCE SYSTEM"
                )

                print(
                    "=" * 80
                )

                break

            # ----------------------------------------------------
            # Empty input
            # ----------------------------------------------------

            if not question:

                print_warning(
                    "Please enter a question."
                )

                continue

            # ----------------------------------------------------
            # Question answering
            # ----------------------------------------------------

            print_step(
                "Processing contract question"
            )

            response = answer_question(

                question=question,

                sections=sections,

                embedding_model=embedding_model,

                qdrant_client=qdrant_client,

                groq_client=groq_client
            )

            display_qa_answer(
                response
            )

        except ValidationError as exc:

            print_error(
                f"Invalid question: {exc}"
            )

        except KeyboardInterrupt:

            print(
                "\n\nExiting..."
            )

            break

        except Exception as exc:

            print_error(
                f"Question processing failed: {exc}"
            )


# ================================================================
# 30. FIND CONTRACT PDF
# ================================================================

def find_contract_pdf() -> str:
    """
    Find the contract PDF inside demo/.
    """

    demo_directory = Path(
        "demo"
    )

    if not demo_directory.exists():

        raise FileNotFoundError(
            "demo/ directory does not exist."
        )

    pdf_files = sorted(
        demo_directory.glob(
            "*.pdf"
        )
    )

    if not pdf_files:

        raise FileNotFoundError(
            """
No PDF found in demo/.

Expected structure:

project/
│
├── demo/
│   └── contract.pdf
│
├── main.py
└── .env
"""
        )

    if len(pdf_files) > 1:

        print_warning(
            "Multiple PDFs found."
        )

        for pdf in pdf_files:

            print(
                f"  • {pdf}"
            )

        print(
            "\nUsing the first PDF."
        )

    return str(
        pdf_files[0]
    )


# ================================================================
# 31. MAIN APPLICATION
# ================================================================

def main():
    """
    Application entry point.

    This function controls the high-level workflow only.
    Business logic remains inside dedicated functions.
    """

    try:

        # --------------------------------------------------------
        # Initialize
        # --------------------------------------------------------

        (
            groq_client,
            qdrant_client,
            embedding_model
        ) = initialize_clients()

        # --------------------------------------------------------
        # Validate Qdrant
        # --------------------------------------------------------

        validate_qdrant_collection(
            qdrant_client
        )

        # --------------------------------------------------------
        # Locate contract
        # --------------------------------------------------------

        pdf_path = find_contract_pdf()

        print_success(
            f"Contract selected: {pdf_path}"
        )

        # --------------------------------------------------------
        # Build LangGraph
        # --------------------------------------------------------

        print_step(
            "Building LangGraph contract-analysis workflow"
        )

        analysis_app = build_analysis_graph(

            groq_client=groq_client,

            embedding_model=embedding_model,

            qdrant_client=qdrant_client
        )

        print_success(
            "LangGraph compiled successfully"
        )

        # --------------------------------------------------------
        # Full contract analysis
        # --------------------------------------------------------

        report, sections = run_contract_analysis(

            app=analysis_app,

            pdf_path=pdf_path
        )

        if not report:

            return

        # --------------------------------------------------------
        # Display structured report
        # --------------------------------------------------------

        display_report(
            report
        )

        # --------------------------------------------------------
        # Start Q&A
        # --------------------------------------------------------

        start_qa_loop(

            sections=sections,

            groq_client=groq_client,

            embedding_model=embedding_model,

            qdrant_client=qdrant_client
        )

    except Exception as exc:

        print_error(
            f"Application startup failed: {exc}"
        )


# ================================================================
# 32. APPLICATION ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()