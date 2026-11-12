import re


class DocumentPreprocessor:
    SECTION_PATTERN = re.compile(
        r"(?=^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|제\s*\d+\s*[장절]))",
        re.MULTILINE,
    )

    SENTENCE_PATTERN = re.compile(
        r"(?<=[!?])\s+"
        r"|(?<=[가-힣A-Za-z]\.)\s+"
        r"|\n(?=[□■▪●•※①②③④⑤⑥⑦⑧⑨⑩])"
    )

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        deduplicated_lines = []

        for line in lines:
            if not deduplicated_lines or line != deduplicated_lines[-1]:
                deduplicated_lines.append(line)

        return "\n".join(deduplicated_lines)

    @classmethod
    def split_sections(cls, text: str) -> list[str]:
        sections = [
            section.strip()
            for section in cls.SECTION_PATTERN.split(text)
            if section.strip()
        ]

        return sections

    @classmethod
    def split_sentences(cls, text: str) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in cls.SENTENCE_PATTERN.split(text)
            if sentence.strip()
        ]

        return sentences

    @classmethod
    def chunk_text(
            cls,
            text: str,
            max_chars: int = 1200,
            overlap_sentences: int = 2,
    ) -> list[str]:
        sections = cls.split_sections(text)

        chunks: list[str] = []

        for section in sections:
            sentences = cls.split_sentences(section)

            current: list[str] = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence) + 1

                if (
                        current
                        and current_length + sentence_length > max_chars
                ):
                    chunks.append("\n".join(current))

                    current = current[-overlap_sentences:]
                    current_length = sum(
                        len(item) + 1
                        for item in current
                    )

                current.append(sentence)
                current_length += sentence_length

            if current:
                chunks.append("\n".join(current))

        return chunks