# Sentiment Analysis with Twitter Sentiment Dataset

This repository contains a simple Python script to train and run a sentiment analysis model on the Kaggle/Twitter Sentiment dataset (training.1600000.processed.noemoticon.csv).

Dataset notes
- Format: CSV with 6 fields: polarity, id, date, query, user, text.
- Polarity: 0 = negative, 2 = neutral, 4 = positive.
- The training labels were heuristically created from emoticons: tweets with :) treated as positive, :( as negative.

Files
- `main.py`: trainer and predictor CLI.
- `requirements.txt`: Python dependencies.

Quickstart
1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```


2. Train a model:

- Sample 5% (quick):

```bash
python3 main.py --train --data archive-2/training.1600000.processed.noemoticon.csv --sample 0.05 --model-out models/sentiment.joblib
```

- Full dataset (may take long and require significant RAM/CPU):

```bash
python3 main.py --train --data archive-2/training.1600000.processed.noemoticon.csv --model-out models/sentiment.joblib
```

3. Predict:

```bash
python3 main.py --predict "I love this!" "This is terrible." --model models/sentiment.joblib
```

Acknowledgements
- Dataset inspired by the Kaggle Twitter Sentiment dataset.
