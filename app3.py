import json
import torch
import pickle
import random
import numpy as np
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ✅ Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load intents dataset
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

# ✅ Load words and labels
words = pickle.load(open("texts.pkl", "rb"))
classes = pickle.load(open("labels.pkl", "rb"))

# ✅ Label mapping
label_map = {tag: idx for idx, tag in enumerate(classes)}

# ✅ Extract sentences and labels
sentences, labels = [], []
for intent in intents["intents"]:
    tag = intent["tag"]
    if tag in label_map:
        for pattern in intent["patterns"]:
            sentences.append(pattern.lower())  # Ensure lowercase
            labels.append(label_map[tag])

# ✅ Ensure labels are valid
num_labels = len(label_map)
assert max(labels) < num_labels, "Error: Some labels exceed num_labels!"

# ✅ Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ✅ Tokenize data
def tokenize_data(sentences, labels):
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=64, return_tensors="pt")
    return encodings, torch.tensor(labels)

# ✅ Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# ✅ Tokenize data
train_encodings, train_labels = tokenize_data(train_texts, train_labels)
val_encodings, val_labels = tokenize_data(val_texts, val_labels)

# ✅ Create dataset class
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# ✅ Create datasets
train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

# ✅ Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ✅ Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
)
model.to(device)

# ✅ Define optimizer & loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# ✅ Training function
def train_model(model, train_loader, epochs=25):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ✅ Train the model
train_model(model, train_loader, epochs=10)

# ✅ Save the trained model locally
model_path = "intent_classifier.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")

# ✅ Function to predict intent
def predict_intent(text):
    """Predict the intent of the given text using the trained model."""
    model.eval()
    tokens = tokenizer(text.lower(), return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    
    with torch.no_grad():
        output = model(**tokens)
    
    logits = output.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    
    predicted_intent = classes[predicted_label]
    print(f"[DEBUG] Predicted intent: {predicted_intent}")  # ✅ Debugging log
    return predicted_intent

# ✅ Function to get response based on intent
def get_response(intent_tag):
    """Return a random response from the intent's responses."""
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    
    return "I'm not sure how to respond to that."

# ✅ Function to handle chatbot response
def chatbot_response(user_input):
    """Process user input and return an appropriate response."""
    intent = predict_intent(user_input)
    return get_response(intent)

# ✅ Chat loop for user interaction
print("Chatbot is running! Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")
