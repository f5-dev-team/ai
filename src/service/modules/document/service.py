from pathlib import Path

from src.service.modules.document.parser import DocumentParser
from src.service.modules.document.preprocessing import (
    DocumentPreprocessor,
)


class DocumentService:
    def __init__(self):
        self.parser = DocumentParser()
        self.preprocessor = DocumentPreprocessor()

    def process(
            self,
            file_path: str | Path,
    ) -> tuple[str, list[str], list[str]]:
        raw_text = self.parser.parse(file_path)

        cleaned_text = self.preprocessor.clean_text(
            raw_text
        )

        sentences = self.preprocessor.split_sentences(
            cleaned_text
        )

        chunks = self.preprocessor.chunk_text(
            cleaned_text
        )

        return cleaned_text, sentences, chunks