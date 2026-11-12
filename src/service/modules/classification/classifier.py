from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class ScholarshipClassifier:
    def __init__(
            self,
            model_path: str | Path = (
                    "artifacts/models/scholarship-classifier-v0"
            ),
            threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(self.model_path)
        )

        self.model.eval()

    def classify(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.sigmoid(
            outputs.logits
        )[0]

        scores = {}

        for index, probability in enumerate(probabilities):
            label = self.model.config.id2label[index]
            scores[label] = float(probability.item())

        predicted_labels = [
            label
            for label, probability in scores.items()
            if probability >= self.threshold
        ]

        top_label = max(
            scores,
            key=scores.get,
        )

        return {
            "text": text,
            "labels": predicted_labels,
            "scores": scores,
            "top_label": top_label,
            "top_score": scores[top_label],
        }

    def classify_sentences(
            self,
            sentences: list[str],
    ) -> list[dict]:
        return [
            self.classify(sentence)
            for sentence in sentences
        ]