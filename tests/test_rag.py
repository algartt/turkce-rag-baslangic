import tempfile
import unittest
from pathlib import Path

from rag import Chunk, TfidfRetriever, load_chunks, split_text, tokenize


class RagTests(unittest.TestCase):
    def test_turkish_tokenization(self):
        self.assertEqual(tokenize("Büyük Dil Modeli, öğreniyor!"), ["büyük", "dil", "modeli", "öğreniyor"])

    def test_split_text_overlap(self):
        parts = split_text("bir iki üç dört beş altı", chunk_size=4, overlap=2)
        self.assertEqual(parts[0], "bir iki üç dört")
        self.assertEqual(parts[1], "üç dört beş altı")

    def test_relevant_chunk_ranks_first(self):
        chunks = [
            Chunk("rag.txt", "RAG güvenilir belgelerden ilgili bilgi getirir"),
            Chunk("vision.txt", "Görüntü sınıflandırma piksel verilerini işler"),
        ]
        result = TfidfRetriever(chunks).search("RAG belge arama", top_k=1)
        self.assertEqual(result[0].chunk.source, "rag.txt")
        self.assertGreater(result[0].score, 0)

    def test_loads_supported_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bilgi.md").write_text("Türkçe yapay zekâ belgesi", encoding="utf-8")
            (root / "ignore.csv").write_text("yok", encoding="utf-8")
            chunks = load_chunks(root)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source, "bilgi.md")


if __name__ == "__main__":
    unittest.main()
