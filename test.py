import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os

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
        
        # Initialize emotions list
        self.emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        self.mlb.fit([self.emotions])

    def load_model(self, path):
        """Load the model state"""
    # Load checkpoint dictionary
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
    
    # Debug: Print checkpoint contents
        print("Checkpoint keys:", checkpoint.keys())
        if "mlb_classes" in checkpoint:
            print("mlb_classes type:", type(checkpoint["mlb_classes"]))
            print("mlb_classes shape or length:", 
                len(checkpoint["mlb_classes"]) if isinstance(checkpoint["mlb_classes"], list) else checkpoint["mlb_classes"].shape)
        
        # IMPORTANT: Use the entire array from the checkpoint
        self.emotions = list(checkpoint["mlb_classes"])
        print(f"Using {len(self.emotions)} emotions from checkpoint")
        
        # Re-fit the MultiLabelBinarizer with the loaded classes
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.emotions])
    
    # Now initialize the model with the correct number of emotions
        self.model = BertEmotionClassifier(len(self.emotions))
    
    # Load the model weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    # Load tokenizer if available
        tokenizer_path = os.path.join(os.path.dirname(path), 'tokenizer')
        if os.path.exists(tokenizer_path):
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
        self.model.to(self.device)
        print(f"Model loaded from {path} with {len(self.emotions)} emotions")
        

    def predict(self, texts, threshold=0.3):
        """Predict emotions for given texts"""
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            
        dataset = EmotionDataset(texts, np.zeros((len(texts), len(self.emotions))), 
                               self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16)
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = (outputs >= threshold).cpu().numpy()
                all_predictions.append(predictions)

        # Concatenate all predictions
        all_predictions = np.vstack(all_predictions)
        
        # Convert to emotion labels
        emotion_predictions = self.mlb.inverse_transform(all_predictions)
        return emotion_predictions

    def predict_proba(self, texts):
        """Predict emotion probabilities for given texts"""
        if isinstance(texts, str):
            texts = [texts]
            
        dataset = EmotionDataset(texts, np.zeros((len(texts), len(self.emotions))), 
                               self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16)
        
        self.model.eval()
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                all_probabilities.append(outputs.cpu().numpy())

        # Concatenate all probabilities
        all_probabilities = np.vstack(all_probabilities)
        
        return all_probabilities, self.emotions

def analyze_text(text, classifier, top_k=5):
    """
    Analyze emotions in a text and return top k emotions with their probabilities
    """
    probabilities, emotions = classifier.predict_proba([text])
    predictions = classifier.predict([text])
    
    # Get all emotions with their probabilities
    emotion_probs = list(zip(emotions, probabilities[0]))
    # Sort by probability
    emotion_probs.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'text': text,
        'predicted_emotions': list(predictions[0]),
        'top_emotions': [
            {'emotion': emotion, 'probability': float(prob)} 
            for emotion, prob in emotion_probs[:top_k]
        ]
    }

def main():
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Load the trained model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'final_model.pt')
    classifier.load_model(model_path)
    
    # Example texts for prediction
    example_texts = [
        "This is absolutely amazing! I can't believe how well it turned out!",
        "I'm feeling quite frustrated and disappointed with the results.",
        "The situation makes me both anxious and hopeful."
    ]
    
    # Analyze each text
    print("/nEmotion Analysis Results:")
    print("-" * 50)
    
    for text in example_texts:
        results = analyze_text(text, classifier)
        
        print(f"\nText: {results['text']}")
        print(f"Predicted Emotions: {', '.join(results['predicted_emotions'])}")
        print("\nTop 5 emotions with probabilities:")
        for emotion in results['top_emotions']:
            print(f"{emotion['emotion']}: {emotion['probability']:.3f}")
        print("-" * 50)
    
    # Interactive mode
    print("\nInteractive Mode (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        text = input("\nEnter text to analyze: ")
        if text.lower() == 'quit':
            break
            
        results = analyze_text(text, classifier)
        print(f"\nmPredicted Emotions: {', '.join(results['predicted_emotions'])}")
        print("\nTop 5 emotions with probabilities:")
        for emotion in results['top_emotions']:
            print(f"{emotion['emotion']}: {emotion['probability']:.3f}")

if __name__ == "__main__":
    main()