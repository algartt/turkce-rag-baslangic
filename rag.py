"""Bağımlılıksız, kaynak gösteren basit bir Türkçe RAG arama motoru."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[\wçğıöşüÇĞİÖŞÜ]+", re.UNICODE)


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    """Türkçe metni küçük harfli sözcük belirteçlerine ayırır."""
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def split_text(text: str, chunk_size: int = 80, overlap: int = 15) -> list[str]:
    """Metni kelime sayısına göre örtüşen parçalara böler."""
    if chunk_size <= 0:
        raise ValueError("chunk_size pozitif olmalıdır")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap, 0 ile chunk_size arasında olmalıdır")

    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step) if words[i : i + chunk_size]]


def load_chunks(directory: Path, chunk_size: int = 80, overlap: int = 15) -> list[Chunk]:
    """Bir klasördeki Markdown ve düz metin belgelerini yükler."""
    chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            relative = str(path.relative_to(directory))
            chunks.extend(Chunk(relative, part) for part in split_text(text, chunk_size, overlap))
    return chunks


class TfidfRetriever:
    """Küçük belge koleksiyonları için bellek içi TF-IDF arayıcı."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("İndekslenecek en az bir belge parçası gereklidir")
        self.chunks = chunks
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        total = len(chunks)
        self.idf = {token: math.log((1 + total) / (1 + count)) + 1 for token, count in document_frequency.items()}
        self.vectors = [self._vector(tokens) for tokens in tokenized]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        return {token: (count / total) * self.idf.get(token, 0.0) for token, count in counts.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        dot = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır")
        query_vector = self._vector(tokenize(query))
        ranked = [SearchResult(chunk, self._cosine(query_vector, vector)) for chunk, vector in zip(self.chunks, self.vectors)]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Yerel ve bağımlılıksız Türkçe RAG araması")
    parser.add_argument("question", help="Belgelerde aranacak soru")
    parser.add_argument("--docs", type=Path, default=Path(__file__).parent / "docs", help="Belge klasörü")
    parser.add_argument("--top-k", type=int, default=3, help="Döndürülecek kaynak sayısı")
    args = parser.parse_args()

    chunks = load_chunks(args.docs)
    results = TfidfRetriever(chunks).search(args.question, args.top_k)
    print(f"Soru: {args.question}\n\nEn ilgili kaynaklar:")
    for index, result in enumerate(results, 1):
        print(f"{index}. [{result.chunk.source} | skor: {result.score:.3f}]\n   {result.chunk.text}\n")


if __name__ == "__main__":
    main()
