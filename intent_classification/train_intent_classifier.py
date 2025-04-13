
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# RoBERTa-based classifier
class RoBERTaIntentClassifier(nn.Module):
    def __init__(self, num_labels):
        super(RoBERTaIntentClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(cls_output)
        logits = self.fc(x)
        return logits

# Dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ✅ Only run training when script is executed directly
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("intent_data1.csv")
    df = df.dropna()

    # Encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])
    num_classes = len(label_encoder.classes_)

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
    )

    # Datasets and loaders
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoBERTaIntentClassifier(num_labels=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 7
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {total_loss / len(train_loader):.4f}")

    # Save model and label encoder
    os.makedirs("intent_classification/model", exist_ok=True)
    torch.save(model.state_dict(), "intent_classification/model/intent_bert.pth")
    torch.save(label_encoder, "intent_classification/model/label_encoder.pt")
