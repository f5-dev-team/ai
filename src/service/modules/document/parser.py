from pathlib import Path

import pymupdf


class DocumentParser:
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        text_parts = []

        with pymupdf.open(path) as document:
            for page in document:
                text = page.get_text("text")
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts)