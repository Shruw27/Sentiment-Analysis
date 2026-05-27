import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import ast
import os
import requests
import zipfile
from tqdm import tqdm

def download_goemotions():
    """
    Download and prepare the GoEmotions dataset
    Returns paths to train, validation, and test files
    """
    base_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    for file in files:
        output_path = os.path.join('data', file)
        if not os.path.exists(output_path):
            print(f"Downloading {file}...")
            response = requests.get(base_url + file)
            with open(output_path, 'wb') as f:
                f.write(response.content)
    
    def process_file(file_path):
        # Read TSV file with correct column names
        df = pd.read_csv(file_path, sep='\t', names=['text', 'emotions', 'id', 'sentiment'])
        
        # Convert comma-separated emotion indices to emotion names
        emotion_mapping = {
            0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 
            4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity', 
            8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust', 
            12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude', 
            16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 
            20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 
            24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
        }
        
        # Convert emotion indices to emotion names
        df['emotion_labels'] = df['emotions'].apply(
            lambda x: [emotion_mapping[int(i)] for i in str(x).split(',')]
        )
        
        return df[['text', 'emotion_labels']]
    
    # Process all files
    print("Processing train.tsv...")
    train_df = process_file('data/train.tsv')
    print("Processing dev.tsv...")
    val_df = process_file('data/dev.tsv')
    print("Processing test.tsv...")
    test_df = process_file('data/test.tsv')
    
    # Save processed files
    train_df.to_csv('data/train_processed.csv', index=False)
    val_df.to_csv('data/val_processed.csv', index=False)
    test_df.to_csv('data/test_processed.csv', index=False)
    
    return 'data/train_processed.csv', 'data/val_processed.csv', 'data/test_processed.csv'

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

class BertEmotionClassifier(nn.Module):
    def __init__(self, n_emotions):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_emotions)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return self.sigmoid(logits)

class EmotionClassifier:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mlb = MultiLabelBinarizer()
        self.model = None
        
        # GoEmotions categories
        self.emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

    def save_model(self, path):
        """Save the model, tokenizer, and label binarizer"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'mlb_classes': self.mlb.classes_
        }, path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(os.path.dirname(path), 'tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
        
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load the model, tokenizer, and label binarizer"""
        # Load tokenizer
        tokenizer_path = os.path.join(os.path.dirname(path), 'tokenizer')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # Load model state and mlb classes
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Initialize model if not already done
        if self.model is None:
            self.model = BertEmotionClassifier(len(self.emotions))
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Restore label binarizer classes
        self.mlb.classes_ = checkpoint['mlb_classes']
        
        print(f"Model loaded from {path}")

    def prepare_data(self, texts, labels=None):
        if labels is not None:
            # Convert string representations of lists to actual lists if needed
            labels = [ast.literal_eval(label) if isinstance(label, str) else label 
                     for label in labels]
            # Transform labels to binary format
            labels = self.mlb.fit_transform(labels)
            return EmotionDataset(texts, labels, self.tokenizer)
        return EmotionDataset(texts, np.zeros((len(texts), len(self.emotions))), 
                            self.tokenizer)

    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16, 
              learning_rate=2e-5, warmup_steps=0):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        if self.model is None:
            self.model = BertEmotionClassifier(len(self.emotions))
            self.model.to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Calculate total training steps for scheduler
        total_steps = len(train_loader) * epochs
        
        # Create scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        criterion = nn.BCELoss()
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({'training_loss': f'{loss.item():.3f}'})

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Average Training Loss: {avg_train_loss:.4f}')
            print(f'Average Validation Loss: {avg_val_loss:.4f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('models/best_model.pt')

    def predict(self, texts, threshold=0.5):
        dataset = self.prepare_data(texts)
        dataloader = DataLoader(dataset, batch_size=16)
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = (outputs >= threshold).cpu().numpy()
                all_predictions.extend(predictions)

        # Convert to emotion labels
        emotion_predictions = self.mlb.inverse_transform(all_predictions)
        return emotion_predictions

def main():
    # Download and prepare dataset
    print("Downloading and preparing GoEmotions dataset...")
    train_path, val_path, test_path = download_goemotions()
    
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Prepare datasets
    train_dataset = classifier.prepare_data(train_df['text'], train_df['emotion_labels'])
    val_dataset = classifier.prepare_data(val_df['text'], val_df['emotion_labels'])
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train model with learning rate scheduling
    print("Training model...")
    classifier.train(
        train_dataset, 
        val_dataset,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        warmup_steps=100
    )
    
    # Save final model
    classifier.save_model('models/final_model.pt')
    
    # Test loading model
    new_classifier = EmotionClassifier()
    new_classifier.load_model('models/final_model.pt')
    
    # Make predictions
    test_texts = ["This is amazing! I love it!", 
                  "I'm feeling quite disappointed and angry about this situation."]
    predictions = new_classifier.predict(test_texts)
    
    print("\nTest predictions:")
    for text, pred in zip(test_texts, predictions):
        print(f"\nText: {text}")
        print(f"Predicted emotions: {pred}")

if __name__ == "__main__":
    main()  