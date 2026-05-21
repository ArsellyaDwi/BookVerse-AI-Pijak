---
title: {{title}}
emoji: {{emoji}}
colorFrom: {{colorFrom}}
colorTo: {{colorTo}}
sdk: {{sdk}}
sdk_version: "{{sdkVersion}}"
{{#pythonVersion}}
python_version: "{{pythonVersion}}"
{{/pythonVersion}}
app_file: app.py
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
