from pathlib import Path

from src.service.modules.classification.classifier import (
    ScholarshipClassifier,
)
from src.service.modules.document.service import (
    DocumentService,
)


CONDITION_LABELS = [
    "ACADEMIC",
    "FINANCIAL",
    "PERSONAL_ELIGIBILITY",
    "ACTIVITY",
    "ADMINISTRATIVE",
    "PERIOD",
]


class DocumentPipeline:
    def __init__(self):
        self.document_service = DocumentService()
        self.classifier = ScholarshipClassifier()

    def run(
            self,
            file_path: str | Path,
            allow_fallback: bool = False,
            fallback_limit: int = 5,
    ) -> dict:
        (
            cleaned_text,
            sentences,
            chunks,
        ) = self.document_service.process(file_path)

        classifications = (
            self.classifier.classify_sentences(sentences)
        )

        candidates = []

        for result in classifications:
            if result["labels"]:
                candidate = dict(result)
                candidate["selection_type"] = "threshold"
                candidates.append(candidate)

        if allow_fallback and not candidates:
            fallback_candidates = []

            for result in classifications:
                condition_score = max(
                    result["scores"][label]
                    for label in CONDITION_LABELS
                )

                candidate = dict(result)
                candidate["selection_type"] = "fallback"
                candidate["fallback_score"] = condition_score

                fallback_candidates.append(candidate)

            candidates = sorted(
                fallback_candidates,
                key=lambda item: item["fallback_score"],
                reverse=True,
            )[:fallback_limit]

        return {
            "text": cleaned_text,
            "sentences": sentences,
            "chunks": chunks,
            "classifications": classifications,
            "candidates": candidates,
        }