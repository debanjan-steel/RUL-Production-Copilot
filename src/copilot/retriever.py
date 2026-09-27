"""BM25 Information Retrieval Engine for Turbofan Maintenance Knowledge Base.

Parses domain engineering markdown documents and provides fast, deterministic
lexical retrieval for grounded root cause analysis and maintenance recommendations.
"""

from pathlib import Path
import re


class KnowledgeBaseRetriever:
    """Parses markdown knowledge base documents and performs BM25 retrieval."""

    def __init__(self, kb_dir: Path | None = None):
        if kb_dir is None:
            root_dir = Path(__file__).resolve().parent.parent.parent
            kb_dir = root_dir / "kb"
        self.kb_dir = kb_dir
        self.documents: list[dict[str, str]] = []
        self._load_and_chunk_documents()
        self._build_index()

    def _load_and_chunk_documents(self) -> None:
        """Splits markdown files into distinct failure mode and sensor chunks."""
        for md_file in self.kb_dir.glob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            # Split sections by markdown level-2 headers
            sections = re.split(r"\n(?=##\s+)", text)
            for sec in sections:
                sec = sec.strip()
                if not sec or sec.startswith("# "):
                    continue
                # Extract header title
                header_match = re.match(r"^##\s+([^\n]+)", sec)
                title = header_match.group(1).strip() if header_match else "General Knowledge"
                self.documents.append({
                    "source": md_file.name,
                    "title": title,
                    "content": sec,
                })

    def _build_index(self) -> None:
        """Builds token index using rank_bm25 with a lightweight internal fallback."""
        self.tokenized_corpus = [
            self._tokenize(doc["title"] + " " + doc["content"]) for doc in self.documents
        ]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self.use_rank_bm25 = True
        except ImportError:
            self.use_rank_bm25 = False

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple alphanumeric tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve(self, query: str, top_k: int = 2) -> list[dict[str, str]]:
        """Retrieves top_k relevant knowledge base sections matching query.
        
        Args:
            query: Fault symptoms, sensor tags (e.g. 'T30 Ps30 HPC fouling').
            top_k: Number of reference snippets to return.
        """
        if not self.documents:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return self.documents[:top_k]

        if self.use_rank_bm25:
            scores = self.bm25.get_scores(tokens)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [self.documents[idx] for idx in ranked_indices[:top_k] if scores[idx] > 0] or self.documents[:top_k]
        else:
            # Deterministic word overlap fallback
            def score_doc(doc_tokens: list[str]) -> int:
                token_set = set(doc_tokens)
                return sum(1 for t in tokens if t in token_set)

            ranked = sorted(self.documents, key=lambda d: score_doc(self._tokenize(d["content"])), reverse=True)
            return ranked[:top_k]
