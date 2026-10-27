
import argparse
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_data(path, sample=None):
	names = ["polarity", "id", "date", "query", "user", "text"]
	# Resolve common locations if the provided path doesn't exist
	if not os.path.exists(path):
		alt = os.path.join("archive-2", path)
		if os.path.exists(alt):
			path = alt
		else:
			# try file in same dir as script
			base_alt = os.path.join(os.path.dirname(__file__), path)
			if os.path.exists(base_alt):
				path = base_alt
			else:
				alt2 = os.path.join(os.path.dirname(__file__), "archive-2", path)
				if os.path.exists(alt2):
					path = alt2
				else:
					# helpful error listing candidate files
					candidates = []
					if os.path.isdir("archive-2"):
						candidates = os.listdir("archive-2")[:20]
					raise FileNotFoundError(
						f"Data file not found: {path!s}.\nTried current path and archive-2/.\n"
						f"If your file lives in the archive-2 folder, pass archive-2/<filename>.\n"
						f"archive-2 contains: {candidates}"
					)

	df = pd.read_csv(path, encoding="latin-1", header=None, names=names)
	if sample is not None:
		if 0 < sample < 1:
			df = df.sample(frac=sample, random_state=42)
		else:
			df = df.sample(n=int(sample), random_state=42)
	return df


def preprocess_text(s: str) -> str:
	s = str(s)
	s = re.sub(r"http\S+", "", s)
	s = re.sub(r"@\w+", "", s)
	s = re.sub(r"#", "", s)
	s = re.sub(r"[^\w\s']", "", s)
	s = s.lower().strip()
	return s


def build_and_train(df, model_out, test_size=0.2):
	# Map labels: keep only negative (0) and positive (4)
	df = df[df.polarity.isin([0, 4])].copy()
	df["label"] = df.polarity.map({0: 0, 4: 1})
	df["clean_text"] = df.text.map(preprocess_text)

	X = df.clean_text.values
	y = df.label.values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

	vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
	X_train_t = vect.fit_transform(X_train)

	clf = LogisticRegression(max_iter=1000)
	clf.fit(X_train_t, y_train)

	X_test_t = vect.transform(X_test)
	preds = clf.predict(X_test_t)

	print("Accuracy:", accuracy_score(y_test, preds))
	print(classification_report(y_test, preds, target_names=["negative", "positive"]))

	# Save model and vectorizer together
	os.makedirs(os.path.dirname(model_out), exist_ok=True)
	joblib.dump({"vect": vect, "clf": clf}, model_out)
	print(f"Saved model to {model_out}")


def incremental_train(path, model_out, chunksize=20000, n_features=2 ** 20):
	"""Train using a streaming approach: HashingVectorizer + SGDClassifier.partial_fit.
	This avoids keeping the whole dataset in memory.
	"""
	# Resolve common locations if the provided path doesn't exist
	if not os.path.exists(path):
		alt = os.path.join("archive-2", path)
		if os.path.exists(alt):
			path = alt
		else:
			base_alt = os.path.join(os.path.dirname(__file__), path)
			if os.path.exists(base_alt):
				path = base_alt
			else:
				alt2 = os.path.join(os.path.dirname(__file__), "archive-2", path)
				if os.path.exists(alt2):
					path = alt2
				else:
					raise FileNotFoundError(f"Data file not found: {path!s}. Tried archive-2/ and script directory.")

	hv = HashingVectorizer(n_features=n_features, alternate_sign=False, stop_words='english', ngram_range=(1, 2))

	# use 'log_loss' for probabilistic/logistic loss (name depends on scikit-learn version)
	clf = SGDClassifier(loss='log_loss', max_iter=5)

	first_pass = True
	reader = pd.read_csv(path, encoding='latin-1', header=None, names=["polarity", "id", "date", "query", "user", "text"], chunksize=chunksize)
	for i, chunk in enumerate(reader):
		# keep only binary labels
		chunk = chunk[chunk.polarity.isin([0, 4])].copy()
		if chunk.empty:
			continue
		texts = chunk.text.map(preprocess_text).tolist()
		X = hv.transform(texts)
		y = chunk.polarity.map({0: 0, 4: 1}).values

		if first_pass:
			clf.partial_fit(X, y, classes=np.array([0, 1]))
			first_pass = False
		else:
			clf.partial_fit(X, y)

		if (i + 1) % 5 == 0:
			print(f"Processed {(i+1)*chunksize} rows")

	# Save classifier and hashing params (vectorizer is stateless)
	os.makedirs(os.path.dirname(model_out), exist_ok=True)
	joblib.dump({"clf": clf, "use_hashing": True, "n_features": n_features, "hash_params": {"alternate_sign": False, "ngram_range": (1, 2)}}, model_out)
	print(f"Saved incremental model to {model_out}")


def predict_text(model_path, texts):
	# Try loading the requested model; on failure, attempt fallbacks from models/ directory
	try:
		obj = joblib.load(model_path)
	except Exception as e:
		print(f"Warning: failed loading {model_path!s}: {e}")
		# search models/ for .joblib files, prefer most recently modified
		candidates = []
		if os.path.isdir("models"):
			for fn in os.listdir("models"):
				if fn.endswith('.joblib'):
					candidates.append(os.path.join('models', fn))
		# ensure we also try some common names if present
		common = ["models/sentiment_10pct.joblib", "models/sentiment_sample.joblib", "models/sentiment.joblib"]
		for c in common:
			if c not in candidates and os.path.exists(c):
				candidates.append(c)

		# sort candidates by modification time (newest first)
		candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
		obj = None
		for c in candidates:
			try:
				print(f"Trying fallback model: {c}")
				obj = joblib.load(c)
				model_path = c
				print(f"Loaded fallback model: {c}")
				break
			except Exception as e2:
				print(f"Failed to load {c}: {e2}")
		if obj is None:
			raise RuntimeError(f"Could not load model {model_path} and no fallbacks succeeded")

	clf = obj.get("clf")
	# recreate vectorizer if hashing-based, otherwise use saved vect
	if obj.get("use_hashing"):
		n_features = obj.get("n_features", 2 ** 20)
		hv = HashingVectorizer(n_features=n_features, alternate_sign=obj.get("hash_params", {}).get("alternate_sign", False), stop_words='english', ngram_range=obj.get("hash_params", {}).get("ngram_range", (1,2)))
		vect = hv
	else:
		vect = obj["vect"]
	clean = [preprocess_text(t) for t in texts]
	X = vect.transform(clean)
	preds = clf.predict(X)
	return ["positive" if p == 1 else "negative" for p in preds]


def main():
	parser = argparse.ArgumentParser(description="Simple sentiment analysis trainer/predictor")
	parser.add_argument("--data", help="Path to training CSV", default="archive-2/training.1600000.processed.noemoticon.csv")
	parser.add_argument("--train", action="store_true", help="Train model from data")
	parser.add_argument("--incremental", action="store_true", help="Use incremental streaming training (HashingVectorizer + SGDClassifier)")
	parser.add_argument("--model-out", help="Output path for model (joblib)", default="models/sentiment.joblib")
	parser.add_argument("--sample", type=float, default=None, help="Fraction of data to sample (0-1) or integer count. Omit to use the full dataset")
	parser.add_argument("--predict", nargs="*", help="Provide text(s) to predict sentiment for")
	parser.add_argument("--model", help="Model path for prediction", default="models/sentiment.joblib")
	parser.add_argument("--interactive", action="store_true", help="Enter interactive REPL to type sentences for prediction")

	args = parser.parse_args()

	# If no action flags provided, default to interactive mode
	if not (args.train or args.predict or args.interactive):
		args.interactive = True

	if args.train:
		print("Loading data from", args.data)
		if args.incremental:
			# incremental training reads the CSV in chunks itself
			incremental_train(args.data, args.model_out)
		else:
			df = load_data(args.data, sample=args.sample)
			print("Rows loaded:", len(df))
			build_and_train(df, args.model_out)

	if args.predict:
		if not os.path.exists(args.model):
			raise SystemExit(f"Model not found: {args.model}. Train first with --train")
		results = predict_text(args.model, args.predict)
		for t, r in zip(args.predict, results):
			print(f"{r}\t{t}")

	if args.interactive:
		if not os.path.exists(args.model):
			raise SystemExit(f"Model not found: {args.model}. Train first with --train")
		print("Interactive mode. Type a sentence and press Enter (empty line to quit).")
		try:
			while True:
				s = input("Enter sentence: ")
				if not s.strip():
					print("Exiting interactive mode.")
					break
				r = predict_text(args.model, [s])[0]
				print(f"Prediction: {r}")
		except KeyboardInterrupt:
			print('\nInterrupted. Exiting interactive mode.')


if __name__ == "__main__":
	main()
