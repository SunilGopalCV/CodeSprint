# 🏁 Code Sprint - Misinformation & Hate Speech Detection with HateBERT

This repository contains the code for a multi-task classification model that detects **Fake News** and **Hate Speech** in tweets. It was developed as part of a Code Sprint event using the **GroNLP/hateBERT** model.

## 🎯 Objective

The goal is to build a multitask learning model that can:
- Classify whether a tweet contains **Fake News** (0 or 1)
- Detect the presence of **Hate Speech** (0 or 1)

## 📜 Rules

- Input will be a tweet (text)
- Output will be two labels:
  - `Fake`: 0 or 1
  - `Hate`: 0 or 1
- Use open-source models and libraries
- Evaluation metric: Classification Report (Precision, Recall, F1-Score)

---

## 📂 Dataset Links

- [Train_Data.xlsx](./Train_Data.xlsx)
- [Val_Data.xlsx](./Val_Data.xlsx)
- [Test_Data.xlsx](./Test_Data.xlsx)

> **Note**: Use **Google Colab** and set the runtime to **T4 GPU** (`Runtime > Change runtime type > Hardware accelerator > GPU (T4)`)

## 🧱 1. Importing Required Libraries

We start by importing all essential libraries: PyTorch for modeling, Transformers for using HateBERT, scikit-learn for evaluation, and pandas for data manipulation.

```python
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
import pandas as pd
```

## 🧹 2. Reading and Cleaning the Dataset
This cell:

- Loads the Excel files for training and validation.
- Cleans the tweets by removing special characters.
- Handles missing values in the labels (`Fake` and `Hate`).

```python
def clean_text(text):
    return re.sub(r'[^A-Za-z0-9 ]+', '', str(text))

train_df = pd.read_excel("Train_Data.xlsx")
val_df = pd.read_excel("Val_Data.xlsx")

train_df['Tweet'] = train_df['Tweet'].apply(clean_text)
val_df['Tweet'] = val_df['Tweet'].apply(clean_text)

train_df['Fake'] = train_df['Fake'].fillna(0).astype(int)
train_df['Hate'] = train_df['Hate'].fillna(0).astype(int)
val_df['Fake'] = val_df['Fake'].fillna(0).astype(int)
val_df['Hate'] = val_df['Hate'].fillna(0).astype(int)
```

## 🔤 3. Initializing the HateBERT Tokenizer
We use the pre-trained tokenizer from `GroNLP/hateBERT`.

```python
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
```

## 🧺 4. Creating a Custom Dataset Class
This `TweetDataset` class prepares the tweets and corresponding labels into the required format for PyTorch.

```python
class TweetDataset(Dataset):
    def __init__(self, texts, fake_labels, hate_labels, tokenizer, max_len=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
        self.fake_labels = list(fake_labels)
        self.hate_labels = list(hate_labels)

    def __len__(self):
        return len(self.fake_labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels_faux': torch.tensor(self.fake_labels[idx]),
            'labels_hate': torch.tensor(self.hate_labels[idx])
        }
```

### 🧠 5. Defining the HateBERT Multitask Model
This is a PyTorch model with two classification heads—one for each task (Fake, Hate).

```python
class HateBERTMultiTask(nn.Module):
    def __init__(self):
        super(HateBERTMultiTask, self).__init__()
        self.bert = AutoModel.from_pretrained("GroNLP/hateBERT")
        self.dropout = nn.Dropout(0.3)
        self.fake_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.hate_classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        fake_logits = self.fake_classifier(pooled_output)
        hate_logits = self.hate_classifier(pooled_output)
        return fake_logits, hate_logits
```

## 📦 6. Preparing Dataloaders
Converts the datasets into PyTorch DataLoader objects for training and validation.

```python
train_dataset = TweetDataset(train_df['Tweet'], train_df['Fake'], train_df['Hate'], tokenizer)
val_dataset = TweetDataset(val_df['Tweet'], val_df['Fake'], val_df['Hate'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
```

## 🏋️‍♂️ 7. Training the Model
This function handles the model training loop, validation, and early stopping.

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            fake_labels = batch['labels_faux'].to(device)
            hate_labels = batch['labels_hate'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            fake_logits, hate_logits = model(input_ids, attention_mask)

            fake_loss = criterion(fake_logits, fake_labels)
            hate_loss = criterion(hate_logits, hate_labels)
            loss = fake_loss + hate_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_fake_preds, val_hate_preds, val_fake_labels, val_hate_labels = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print("Fake Classification Report:")
        print(classification_report(val_fake_labels, val_fake_preds))
        print("Hate Classification Report:")
        print(classification_report(val_hate_labels, val_hate_preds))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break
```

## 🧪 8. Evaluation Function
Evaluates model performance on validation set using classification reports.

```python
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_fake_preds = []
    all_hate_preds = []
    all_fake_labels = []
    all_hate_labels = []

    with torch.no_grad():
        for batch in val_loader:
            fake_labels = batch['labels_faux'].to(device)
            hate_labels = batch['labels_hate'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            fake_logits, hate_logits = model(input_ids, attention_mask)
            fake_loss = criterion(fake_logits, fake_labels)
            hate_loss = criterion(hate_logits, hate_labels)
            loss = fake_loss + hate_loss
            val_loss += loss.item()

            all_fake_preds.extend(torch.argmax(fake_logits, dim=1).cpu().numpy())
            all_hate_preds.extend(torch.argmax(hate_logits, dim=1).cpu().numpy())
            all_fake_labels.extend(fake_labels.cpu().numpy())
            all_hate_labels.extend(hate_labels.cpu().numpy())

    val_loss /= len(val_loader)
    return val_loss, all_fake_preds, all_hate_preds, all_fake_labels, all_hate_labels
```

## 🚀 9. Model Initialization and Training Start
Instantiate the model, define loss and optimizer, and begin training.

```python
model = HateBERTMultiTask()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
train_model(model, train_loader, val_loader, criterion, optimizer)
```

## 📡 10. Making Predictions on a Single Tweet
Function to predict whether a given tweet is fake and/or hate-filled.

```python
def predict_tweet(model, tweet):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded = tokenizer(tweet, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        fake_logits, hate_logits = model(input_ids, attention_mask)
        fake_pred = torch.argmax(fake_logits, dim=1).item()
        hate_pred = torch.argmax(hate_logits, dim=1).item()

    return fake_pred, hate_pred
```

## 📝 11. Predicting on a Sample Tweet
Example usage of the prediction function.

```python
tweet = "This is a test tweet about politics."
fake, hate = predict_tweet(model, tweet)
print(f"Tweet: {tweet}\nPredicted Fake: {fake}, Predicted Hate: {hate}")
```

📈 12. Final Predictions on Test Dataset
Cleans test data, runs predictions, and saves them to `submit.csv.`

```python
test_df = pd.read_excel("Test_Data.xlsx")
test_df['Tweet'] = test_df['Tweet'].apply(clean_text)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

all_fake_preds = []
all_hate_preds = []

with torch.no_grad():
    for tweet in test_df['Tweet']:
        encoded = tokenizer(tweet, return_tensors='pt', truncation=True, padding=True, max_length=128)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        fake_logits, hate_logits = model(input_ids, attention_mask)
        fake_pred = torch.argmax(fake_logits, dim=1).item()
        hate_pred = torch.argmax(hate_logits, dim=1).item()

        all_fake_preds.append(fake_pred)
        all_hate_preds.append(hate_pred)

test_df['Fake'] = all_fake_preds
test_df['Hate'] = all_hate_preds

test_df.to_csv("submit.csv", index=False)
print("Predictions saved to submit.csv")
```
---

# ✅ Results
Predictions are stored in `submit.csv`

Ready for submission.

# 🙌 Credits
Built using 🤗 Transformers and PyTorch.
