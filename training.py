import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import joblib

# ========================
# CONFIG
# ========================
DATASET_PATH = "./dataset_model/emotion_dataset.csv"
OUTPUT_TRAINED_MODEL_PATH = "./trained_model/model.pth"

MODEL_PATH = "./base_model/distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# LOAD DATA
# ========================
df = pd.read_csv(DATASET_PATH)

df["labels"] = df["labels"].apply(lambda x: x.split("|"))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["labels"])

joblib.dump(mlb, "label_encoder.pkl")

# ========================
# DATASET CLASS
# ========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        inputs.pop("token_type_ids", None)

        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item

dataset = EmotionDataset(df["text"].tolist(), y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

model = EmotionClassifier(MODEL_PATH, y.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# ========================
# TRAINING
# ========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

# ========================
# SAVE
# ========================
torch.save(model.state_dict(), OUTPUT_TRAINED_MODEL_PATH)
print("Model saved!")
