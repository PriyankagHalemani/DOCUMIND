import re

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # Remove null characters or binary garbage
        text = text.replace("\x00", "")

        # Join hyphenated words split across lines: e.g. "super-\nvised" -> "supervised"
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # Replace excessive multiple carriage returns and linefeeds
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)

        # Replace tabs and multiple horizontal spaces with a single space
        text = re.sub(r'[ \t]+', ' ', text)

        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Clean line starts/ends
        lines = [line.strip() for line in text.split('\n')]
        cleaned_text = '\n'.join(lines)

        return cleaned_text.strip()
