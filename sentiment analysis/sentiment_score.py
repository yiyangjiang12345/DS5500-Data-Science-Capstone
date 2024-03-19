from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator
import torch

# Ensure CUDA (GPU support) is available and set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment').to(device)

class ReviewDataset(Dataset):
    def __init__(self, reviews):
        self.encodings = tokenizer(reviews, truncation=True, padding=True, max_length=512, return_tensors="pt")

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def sentiment_score(batch):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        results = model(**batch)
    scores = torch.argmax(results.logits, dim=1).cpu().numpy()
    return scores + 1  # Adjust scores to match your scale

def main():
    df = pd.read_csv('processed_reviews.csv')
    dataset = ReviewDataset(df['review'].tolist())
    loader = DataLoader(dataset, batch_size=16, collate_fn=default_data_collator)

    scores = []
    for batch in loader:
        batch_scores = sentiment_score(batch)
        scores.extend(batch_scores)

    df['sentiment_score'] = scores
    df.to_csv('sentiment_scores.csv', index=False)

if __name__ == '__main__':
    main()



