import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib

TRAINED_MODEL_PATH = "./trained_model/model.pth"
BASE_MODEL_PATH = "./base_model/distilbert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# LOAD LABEL
# ========================
mlb = joblib.load("label_encoder.pkl")

# ========================
# MODEL
# ========================
class EmotionClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled)
        return self.fc(x)

# ========================
# LOAD MODEL
# ========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

model = EmotionClassifier(BASE_MODEL_PATH, len(mlb.classes_))
model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ========================
# PREDICT MULTI LABEL
# ========================
def predict(text, threshold=0.5):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    results = []
    for i, p in enumerate(probs):
        if p >= threshold:
            results.append((mlb.classes_[i], float(p)))

    return results

# ========================
# TEST
# ========================
if __name__ == "__main__":
    text = input("Input text: ")
    result = predict(text)

    print("\nDetected emotions:")
    for emo, score in result:
        print(f"{emo} ({score:.2f})")
