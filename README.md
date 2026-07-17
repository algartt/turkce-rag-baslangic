# Türkçe RAG Başlangıç Projesi

Harici API veya ücretli servis kullanmadan çalışan, sade bir **Retrieval-Augmented Generation (RAG)** başlangıç projesi. Metin belgelerini parçalara ayırır, TF-IDF vektörleriyle indeksler ve soruyla en ilgili bölümleri kaynaklarıyla birlikte getirir.

## Neler öğreneceksiniz?

- Belgeleri anlamlı parçalara ayırma
- TF-IDF tabanlı yerel embedding yaklaşımı
- Kosinüs benzerliğiyle kaynak arama
- Kaynaklı cevap bağlamı oluşturma
- RAG sistemleri için basit değerlendirme

## Hızlı başlangıç

Python 3.10 veya üzeri yeterlidir; ek paket gerekmez.

```bash
python rag.py "RAG sistemi ne işe yarar?"
```

Kendi belgelerinizi kullanmak için `.txt` veya `.md` dosyalarını bir klasöre koyun:

```bash
python rag.py "Model halüsinasyonu nasıl azaltılır?" --docs belgelerim --top-k 3
```

Örnek çıktı:

```text
Soru: RAG sistemi ne işe yarar?

En ilgili kaynaklar:
1. [docs/yapay_zeka.txt | skor: 0.42]
   RAG, bir dil modelinin cevap üretmeden önce güvenilir belgelerden bilgi getirmesini sağlar.
```

## Testler

```bash
python -m unittest discover -s tests -v
```

## Mimari

```text
Belgeler → Parçalama → TF-IDF indeksi → Benzerlik araması → Kaynaklı bağlam
```

Bu proje eğitim amaçlı, şeffaf bir temel sunar. Üretim ortamında semantik embedding modeli, vektör veritabanı, yeniden sıralama ve cevap üretim modeli eklenebilir.

## Lisans

MIT
