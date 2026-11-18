import argparse
import csv
import io
import json
import random
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertForSequenceClassification

from moderation import (
    FILES,
    LABELS,
    MODELS,
    FeedForward,
    average_vectors,
    normalize,
    quantize,
    tokens,
)

DATA_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
BASE_MODEL = "dbmdz/bert-base-turkish-cased"
BASE_REVISION = "b6e1de16c983e0f2c70664591ea3f22810072608"
SEED = 42
MAX_LENGTH = 64


def load_data(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read()
        path.write_bytes(archive_bytes)

    with ZipFile(path) as archive:
        def rows(suffix, delimiter):
            member = next(name for name in archive.namelist() if name.endswith("/" + suffix))
            text = archive.read(member).decode("utf-8-sig")
            return list(csv.reader(io.StringIO(text), delimiter=delimiter, quoting=csv.QUOTE_NONE))

        training = rows("offenseval-tr-training-v1.tsv", "\t")[1:]
        testing = rows("offenseval-tr-testset-v1.tsv", "\t")[1:]
        test_labels = dict(rows("offenseval-tr-labela-v1.tsv", ","))
    labels = {"NOT": 0, "OFF": 1}
    return (
        ([text for _, text, _ in training], [labels[label] for _, _, label in training]),
        ([text for _, text in testing], [labels[test_labels[key]] for key, _ in testing]),
    )


def fit_fnn(texts, labels, path):
    dimensions = 64
    embeddings = Word2Vec([tokens(text) for text in texts], vector_size=dimensions,
                          min_count=2, workers=1, seed=SEED, epochs=10).wv
    vectors = {word: embeddings[word] for word in embeddings.index_to_key}
    features = average_vectors(texts, vectors, dimensions)
    targets = torch.tensor(labels)
    weights = len(labels) / (len(LABELS) * torch.bincount(targets).float())
    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = FeedForward(dimensions)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)
    criterion = nn.CrossEntropyLoss(weight=weights)
    for _ in range(20):
        for batch, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), target)
            loss.backward()
            optimizer.step()
    torch.save({"dimensions": dimensions, "words": embeddings.index_to_key,
                "vectors": torch.from_numpy(embeddings.vectors),
                "model": model.state_dict()}, path)


def fit_bert(texts, labels, args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=BASE_REVISION,
                                              model_max_length=MAX_LENGTH)
    model = BertForSequenceClassification.from_pretrained(
        args.base_model, revision=BASE_REVISION, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
        attn_implementation="eager",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    targets = torch.tensor(labels)
    weights = len(labels) / (len(LABELS) * torch.bincount(targets).float())
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    loader = DataLoader(list(zip(texts, labels)), batch_size=args.batch_size, shuffle=True)
    model.train()
    for epoch in range(args.epochs):
        for step, (batch_text, batch_label) in enumerate(loader, start=1):
            batch = tokenizer(list(batch_text), padding=True, truncation=True,
                              return_tensors="pt").to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(**batch).logits, batch_label.to(device))
            loss.backward()
            optimizer.step()
            if step % 50 == 0 or step == len(loader):
                print(f"BERT epoch {epoch + 1}/{args.epochs}, step {step}/{len(loader)}, "
                      f"loss {loss.item():.4f}", flush=True)
    model.cpu().eval()
    model.config.save_pretrained(args.artifacts / "bert")
    tokenizer.save_pretrained(args.artifacts / "bert")
    torch.save(model.state_dict(), args.artifacts / FILES[MODELS[2]])
    torch.save(quantize(model).state_dict(), args.artifacts / FILES[MODELS[3]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/offenseval.zip"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-limit", type=int, help="Stratified subset for development runs")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch size and threads must be positive")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    (texts, labels), (test_texts, _) = load_data(args.data)
    official_train_size = len(texts)
    if args.train_limit is not None:
        texts, _, labels, _ = train_test_split(
            texts, labels, train_size=args.train_limit, stratify=labels, random_state=SEED,
        )
    args.artifacts.mkdir(parents=True, exist_ok=True)
    test_keys = {normalize(text) for text in test_texts}
    normalized = [(normalize(text), label) for text, label in zip(texts, labels)]
    retained = [(text, label) for text, label in normalized if text not in test_keys]
    texts, labels = map(list, zip(*retained))
    print(f"Training on {len(texts)} messages; held-out test contains {len(test_texts)}",
          flush=True)
    baseline = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                             min_df=2, max_features=80_000),
                             LogisticRegression(class_weight="balanced", max_iter=500))
    baseline.fit(texts, labels)
    joblib.dump(baseline, args.artifacts / FILES[MODELS[0]], compress=3)
    print("Training Word2Vec and feedforward network", flush=True)
    fit_fnn(texts, labels, args.artifacts / FILES[MODELS[1]])
    print("Fine-tuning BERTurk", flush=True)
    fit_bert(texts, labels, args)
    metadata = {"dataset": DATA_URL, "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION, "seed": SEED,
                "official_train_size": official_train_size, "train_size": len(texts),
                "removed_test_overlaps": len(normalized) - len(retained),
                "test_size": len(test_texts), "bert_epochs": args.epochs,
                "bert_max_tokens": MAX_LENGTH, "bert_batch_size": args.batch_size}
    (args.artifacts / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()