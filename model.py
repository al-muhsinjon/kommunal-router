"""TF-IDF and full-batch softmax regression using only NumPy."""
from collections import Counter
import numpy as np
from text import is_billing, normalize

LABELS = np.array(["billing", "electricity", "gas", "waste", "water"])


def features(text):
    words = normalize(text).split()
    counts = Counter("w:" + word for word in words)
    for word in words:
        padded = "_" + word + "_"
        counts.update("c:" + padded[i:i + n] for n in (3, 4, 5)
                      for i in range(len(padded) - n + 1))
    return counts


class Tfidf:
    def fit(self, texts):
        df = Counter()
        for text in texts:
            df.update(features(text).keys())  # Each feature once per document.
        self.vocabulary = {term: i for i, term in enumerate(sorted(df))}
        self.idf = np.array([np.log((1 + len(texts)) / (1 + df[term])) + 1
                             for term in self.vocabulary])
        return self

    def transform(self, texts):
        x = np.zeros((len(texts), len(self.vocabulary)))
        for row, text in enumerate(texts):
            for term, count in features(text).items():
                if term in self.vocabulary:
                    x[row, self.vocabulary[term]] = 1 + np.log(count)
        x *= self.idf
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def loss_gradient(x, y, weights, bias, l2):
    logits = x @ weights + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(len(y)), y].mean() + l2 * (weights ** 2).sum() / 2
    residual = np.exp(log_probs)
    residual[np.arange(len(y)), y] -= 1
    residual /= len(y)
    return float(loss), x.T @ residual + l2 * weights, residual.sum(axis=0)


class Softmax:
    def fit(self, x, y, lr=1.0, epochs=600, l2=0.001):
        self.weights = np.zeros((x.shape[1], len(LABELS)))
        self.bias = np.zeros(len(LABELS))
        self.losses = []
        for _ in range(epochs):
            loss, dw, db = loss_gradient(x, y, self.weights, self.bias, l2)
            self.losses.append(loss)
            self.weights -= lr * dw
            self.bias -= lr * db
        self.losses.append(loss_gradient(x, y, self.weights, self.bias, l2)[0])
        return self

    def predict(self, x):
        return LABELS[(x @ self.weights + self.bias).argmax(axis=1)]


def route(model, vectorizer, texts):
    predictions = model.predict(vectorizer.transform(texts))
    return np.array(["billing" if is_billing(t) else p for t, p in zip(texts, predictions)])


def save(path, vectorizer, model):
    np.savez_compressed(path, vocabulary=np.array(list(vectorizer.vocabulary)),
                        idf=vectorizer.idf, weights=model.weights, bias=model.bias)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        vectorizer, model = Tfidf(), Softmax()
        vectorizer.vocabulary = {term: i for i, term in enumerate(data["vocabulary"])}
        vectorizer.idf = data["idf"]
        model.weights, model.bias = data["weights"], data["bias"]
    return vectorizer, model
