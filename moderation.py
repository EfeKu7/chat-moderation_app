import re
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

LABELS = ("CLEAN", "OFFENSIVE")
MODELS = ("TF-IDF", "Word2Vec + FNN", "BERTurk FP32", "BERTurk INT8")
FILES = dict(zip(MODELS, ("tfidf.joblib", "word2vec_fnn.pt", "bert_fp32.pt", "bert_int8.pt")))
LEXICON = {"salak", "aptal", "gerzek", "ahmak", "ezik", "kafasız", "şapşal", "şerefsiz"}
TURKISH_LOWER = str.maketrans("Iİ", "ıi")
TURKISH_ASCII = str.maketrans("çğıöşü", "cgiosu")
LEETSPEAK = str.maketrans("013457@$", "oieastas")


def normalize(text):
    text = text.translate(TURKISH_LOWER).lower()
    text = re.sub(r"https?://\S+|(?<!\w)url\b", " url ", text)
    text = re.sub(r"@user\b", " user ", text)
    text = re.sub(r"[\w@$]+", lambda match: match[0].translate(LEETSPEAK)
                  if re.search(r"[^\W\d_]", match[0]) else match[0], text)
    text = re.sub(r"([^\W\d_])\1{2,}", r"\1", text)
    return " ".join(text.split())


def tokens(text):
    return re.findall(r"[^\W_]+", text)


def edit_distance(left, right):
    row = list(range(len(right) + 1))
    for index, first in enumerate(left, start=1):
        next_row = [index]
        for column, second in enumerate(right, start=1):
            next_row.append(min(row[column] + 1, next_row[-1] + 1,
                                row[column - 1] + (first != second)))
        row = next_row
    return row[-1]


def lexicon_flags(text):
    words = {word.translate(TURKISH_ASCII) for word in tokens(text)}
    return any(
        abs(len(word) - len(entry)) <= 1
        and edit_distance(word, entry) <= (1 if len(entry) >= 5 else 0)
        for word in words for entry in ASCII_LEXICON
    )


ASCII_LEXICON = {word.translate(TURKISH_ASCII) for word in LEXICON}


class FeedForward(nn.Sequential):
    def __init__(self, dimensions):
        super().__init__(nn.Linear(dimensions, 64), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(64, len(LABELS)))


def average_vectors(texts, vectors, dimensions):
    features = np.zeros((len(texts), dimensions), dtype=np.float32)
    for index, text in enumerate(texts):
        found = [vectors[word] for word in tokens(text) if word in vectors]
        if found:
            features[index] = np.mean(found, axis=0)
    return torch.from_numpy(features)


def quantize(model):
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear},
                                                  dtype=torch.qint8)


class Moderator:
    def __init__(self, model_name, artifacts):
        self.name = model_name
        self.artifacts = Path(artifacts)
        path = self.artifacts / FILES[model_name]
        if model_name == MODELS[0]:
            self.model = joblib.load(path)
            return

        state = torch.load(path, map_location="cpu", weights_only=True)
        if model_name == MODELS[1]:
            self.dimensions = state["dimensions"]
            self.vectors = {word: vector.numpy() for word, vector in
                            zip(state["words"], state["vectors"])}
            self.model = FeedForward(self.dimensions)
            self.model.load_state_dict(state["model"])
        else:
            metadata = self.artifacts / "bert"
            self.tokenizer = AutoTokenizer.from_pretrained(metadata, local_files_only=True)
            self.max_length = self.tokenizer.model_max_length
            self.model = BertForSequenceClassification(BertConfig.from_pretrained(metadata))
            if model_name == MODELS[3]:
                self.model = quantize(self.model)
            self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def probabilities(self, messages):
        texts = [normalize(message) for message in messages]
        if self.name == MODELS[0]:
            return self.model.predict_proba(texts)
        if self.name == MODELS[1]:
            logits = self.model(average_vectors(texts, self.vectors, self.dimensions))
        else:
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                    padding=True, max_length=self.max_length)
            logits = self.model(**inputs).logits
        return logits.softmax(dim=-1).numpy()

    def classify(self, message):
        probabilities = self.probabilities([message])[0]
        index = int(probabilities.argmax())
        return LABELS[index], float(probabilities[index])