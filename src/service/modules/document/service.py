from src.service.modules.document.parser import DocumentParser
from src.service.modules.document.preprocessing import DocumentPreprocessor


class DocumentService:
    def __init__(self):
        self.parser = DocumentParser()
        self.preprocessor = DocumentPreprocessor()

    def process_pdf(self, file_path: str) -> list[str]:
        raw_text = self.parser.parse_pdf(file_path)

        cleaned_text = self.preprocessor.clean_text(raw_text)

        chunks = self.preprocessor.chunk_text(cleaned_text)

        return chunks