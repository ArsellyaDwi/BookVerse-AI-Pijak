---
title: BookVerse AI Personality
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# setup installasi

## 1. masukkan dataset pada folder dataset

format: csv
[labels, text]

## 2. jalankan perintah ini

1.
```bash
cd base_model
winget install git-xet
git clone https://huggingface.co/distilbert/distilbert-base-uncased
```

## 3. Jalankan training dengan 

```bash
python training.py
```

## 4. Jalankan prediksi dengan

```bash
python run.py
```
