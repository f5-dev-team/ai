from pathlib import Path

import pymupdf
from extract_hwp import extract_text_from_hwp


class DocumentParser:
    @staticmethod
    def parse(file_path: str | Path) -> str:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return DocumentParser.parse_pdf(path)

        if suffix in {".hwp", ".hwpx"}:
            return DocumentParser.parse_hwp(path)

        raise ValueError(f"지원하지 않는 문서 형식입니다: {suffix}")

    @staticmethod
    def parse_pdf(file_path: str | Path) -> str:
        path = Path(file_path)
        text_parts: list[str] = []

        with pymupdf.open(path) as document:
            for page in document:
                text = page.get_text("text")

                if text:
                    text_parts.append(text)

        return "\n".join(text_parts).strip()

    @staticmethod
    def parse_hwp(file_path: str | Path) -> str:
        text, error = extract_text_from_hwp(str(file_path))

        if error is not None:
            raise ValueError(
                f"HWP 텍스트 추출에 실패했습니다: {error}"
            )

        return text.strip()