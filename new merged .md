import re
import unicodedata
from typing import List, Tuple, Dict, Optional
import pandas as pd

# -----------------------
# TextPreprocessor Class
# -----------------------
class TextPreprocessor:
    def __init__(self):
        # Regex patterns
        self.re_email = re.compile(r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})")
        self.re_url = re.compile(r"\bhttps?://\S+|\bwww\.\S+\b", re.IGNORECASE)
        self.re_phone = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?[\d\-.\s]{6,15}\d")
        self.re_ip = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        self.re_credit = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
        self.re_ssn = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        self.re_datetime = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b")
        self.re_time = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
        self.re_punct = re.compile(r"[^\w\s-]")
        self.re_nonprint = re.compile(r"[\x00-\x1f\x7f-\x9f]")

        self._mask_tokens = {
            "email": "<EMAIL>",
            "phone": "<PHONE>",
            "url": "<URL>",
            "ip": "<IP>",
            "credit": "<CREDIT>",
            "ssn": "<SSN>",
            "date": "<DATE>",
            "time": "<TIME>",
        }

    def normalize_unicode(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        return self.re_nonprint.sub(" ", text)

    def preprocess_text(self, text: Optional[str],
                        lower: bool = True,
                        mask_emails: bool = True,
                        mask_phones: bool = True,
                        mask_urls: bool = True,
                        mask_ip: bool = True,
                        mask_credit: bool = True,
                        mask_ssn: bool = True,
                        remove_dates: bool = True,
                        remove_times: bool = True,
                        remove_punctuation: bool = False,
                        ) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        text = self.normalize_unicode(text)

        if mask_urls:
            text = self.re_url.sub(self._mask_tokens["url"], text)
        if mask_emails:
            text = self.re_email.sub(self._mask_tokens["email"], text)
        if mask_phones:
            text = self.re_phone.sub(self._mask_tokens["phone"], text)
        if mask_ip:
            text = self.re_ip.sub(self._mask_tokens["ip"], text)
        if mask_credit:
            text = self.re_credit.sub(self._mask_tokens["credit"], text)
        if mask_ssn:
            text = self.re_ssn.sub(self._mask_tokens["ssn"], text)
        if remove_dates:
            text = self.re_datetime.sub(self._mask_tokens["date"], text)
        if remove_times:
            text = self.re_time.sub(self._mask_tokens["time"], text)
        if remove_punctuation:
            text = self.re_punct.sub(" ", text)

        text = re.sub(r"\s{2,}", " ", text).strip()
        if lower:
            text = text.lower()
        return text

    def preprocess_dataframe(self, df: pd.DataFrame,
                             columns: List[str],
                             **kwargs) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[int, str, str]]]]:
        df = df.copy()
        replacements_log = {col: [] for col in columns}

        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not in DataFrame")

            new_col = f"{col}_cleaned"
            df[new_col] = ""

            for idx, orig in df[col].items():
                try:
                    processed = self.preprocess_text(orig, **kwargs)
                except Exception:
                    processed = orig
                df.at[idx, new_col] = processed
                replacements_log[col].append((idx, orig, processed))

        return df, replacements_log


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    demo = pd.DataFrame({
        "id": [1, 2, 3],
        "text": [
            "Email me at test@example.com and visit http://example.com on 2023-08-12",
            "Phone +91 98765 43210 and SSN 123-45-6789 at 10:34",
            "Credit card 4111 1111 1111 1111 and IP 192.168.0.1"
        ]
    })

    tp = TextPreprocessor()

    processed_df, log = tp.preprocess_dataframe(
        demo,
        columns=["text"],
        lower=False,
        mask_emails=True,
        mask_phones=True,
        mask_urls=True,
        mask_ip=True,
        mask_credit=True,
        mask_ssn=True,
        remove_dates=True,
        remove_times=True,
        remove_punctuation=False
    )

    # Save both original + cleaned columns
    processed_df.to_csv("processed_output.csv", index=False)

    print(processed_df)