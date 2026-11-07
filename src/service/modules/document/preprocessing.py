import re


class DocumentPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        # 연속 공백 정리
        text = re.sub(r"[ \t]+", " ", text)

        # 불필요한 빈 줄 정리
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 각 줄 앞뒤 공백 제거
        lines = [line.strip() for line in text.splitlines()]

        return "\n".join(line for line in lines if line)

    @staticmethod
    def chunk_text(
            text: str,
            max_chars: int = 1000,
            overlap_chars: int = 150,
    ) -> list[str]:

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n")
            if paragraph.strip()
        ]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            candidate = (
                f"{current_chunk}\n{paragraph}"
                if current_chunk
                else paragraph
            )

            if len(candidate) <= max_chars:
                current_chunk = candidate
                continue

            if current_chunk:
                chunks.append(current_chunk)

                overlap = current_chunk[-overlap_chars:]
                current_chunk = f"{overlap}\n{paragraph}"
            else:
                # 한 문단 자체가 너무 긴 경우
                for start in range(
                        0,
                        len(paragraph),
                        max_chars - overlap_chars,
                ):
                    chunks.append(
                        paragraph[start:start + max_chars]
                    )

                current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks