import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BERT_MODEL_NAME = 'bert-base-uncased'
TEXT_MODEL_PATH = "models/checkpoints/text_model.pth"
IMAGE_MODEL_PATH = "models/checkpoints/image_model.pth"

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_text_model():
    print("--- Training Text Model for Fake News Detection ---")
    texts = ["This is a real news article about science.", "This is fake news, a politician said something."] * 10
    labels = [0, 1] * 10
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)
    model.to(device)
    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    print("Starting text model training...")
    for epoch in range(1):
        for batch in train_dataloader:
            model.train()
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    print("Text model training complete. Saving checkpoint...")
    os.makedirs(os.path.dirname(TEXT_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), TEXT_MODEL_PATH)
    print(f"Text model saved to {TEXT_MODEL_PATH}")

class DeepfakeImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, item):
        image = Image.new('RGB', (224, 224), color = 'red')
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, torch.tensor(label, dtype=torch.long)

def train_image_model():
    print("\n--- Training Image Model for Deepfake Detection ---")
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"] * 10
    labels = [0, 1] * 10
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    train_dataset = DeepfakeImageDataset(train_paths, train_labels, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Starting image model training...")
    for epoch in range(1):
        for inputs, labels in train_dataloader:
            model.train()
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
    print("Image model training complete. Saving checkpoint...")
    os.makedirs(os.path.dirname(IMAGE_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), IMAGE_MODEL_PATH)
    print(f"Image model saved to {IMAGE_MODEL_PATH}")

if __name__ == '__main__':
    train_text_model()
    train_image_model()
