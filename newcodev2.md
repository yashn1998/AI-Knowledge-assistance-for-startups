Nice — I updated your script to remove stopwords and person names using spaCy (preferably the transformer model en_core_web_trf) and to keep the cleaned columns next to the originals in the saved CSV. The new code:

Tries to load en_core_web_trf by default; if unavailable it falls back to en_core_web_sm and prints a warning.

Removes (optionally) spaCy stopwords and tokens that belong to PERSON named-entity spans.

Makes spaCy-based removal configurable (use_spacy, spacy_model, remove_spacy_stopwords, remove_persons) so you can turn it off for speed or testing.

Keeps your existing masking (emails/URLs/phones/etc.) and punctuation/date/time removal options, but runs the spaCy-based removals before the regex masking (so NER sees the raw tokens).

Preserves the CSV output behavior: original column + cleaned column inserted next to it.


Copy/paste the complete updated script below.

import re
import unicodedata
from typing import List, Tuple, Dict, Optional
import pandas as pd

# Try to import spaCy, but keep script usable if spaCy isn't installed.
try:
    import spacy
except Exception:
    spacy = None

# -----------------------
# TextPreprocessor Class
# -----------------------
class TextPreprocessor:
    def __init__(
        self,
        spacy_model: str = "en_core_web_trf",
        use_spacy: bool = True,
        remove_spacy_stopwords: bool = True,
        remove_persons: bool = True,
    ):
        """
        spacy_model: model name to load (default: en_core_web_trf).
        use_spacy: whether to attempt spaCy processing (NER + stopwords).
        remove_spacy_stopwords: whether to remove stopwords using spaCy's token.is_stop.
        remove_persons: whether to remove tokens that are part of PERSON named-entity spans.
        """
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

        # spaCy settings
        self.use_spacy = use_spacy and (spacy is not None)
        self.spacy_model_name = spacy_model
        self.remove_spacy_stopwords = remove_spacy_stopwords
        self.remove_persons = remove_persons
        self.nlp = None

        if self.use_spacy:
            try:
                # Try loading the requested model
                self.nlp = spacy.load(self.spacy_model_name)
            except Exception as e:
                # Fallback to small model if trf not available; warn user.
                try:
                    if spacy is not None:
                        self.nlp = spacy.load("en_core_web_sm")
                        print(
                            f"[warning] failed to load '{self.spacy_model_name}'. "
                            "Falling back to 'en_core_web_sm'.\n"
                            "To use the transformer model install it with:\n"
                            "  pip install -U 'spacy[transformers]' && python -m spacy download en_core_web_trf"
                        )
                    else:
                        print("[warning] spaCy is not installed; spaCy-based removals disabled.")
                        self.use_spacy = False
                except Exception:
                    # If even small model isn't available, disable spaCy
                    print("[warning] Could not load spaCy model. spaCy-based removals disabled.")
                    self.use_spacy = False
                    self.nlp = None

    def normalize_unicode(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        return self.re_nonprint.sub(" ", text)

    def _apply_spacy_removals(self, text: str) -> str:
        """
        Run spaCy pipeline to remove stopwords and PERSON names as configured.
        Runs only if self.use_spacy and self.nlp are available.
        Returns the filtered text (string).
        """
        if not self.use_spacy or self.nlp is None or not text:
            return text

        try:
            doc = self.nlp(text)
        except Exception:
            # In case spaCy fails on specific text (very long), just return original
            return text

        # Collect token indices part of PERSON entities (if enabled)
        person_token_idxs = set()
        if self.remove_persons:
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    for t in ent:
                        person_token_idxs.add(t.i)

        # Build token list removing tokens as requested
        out_tokens: List[str] = []
        for token in doc:
            if token.i in person_token_idxs:
                # drop person tokens
                continue
            if self.remove_spacy_stopwords and token.is_stop:
                continue
            # Keep token text (use token.text to avoid additional whitespace tokens)
            out_tokens.append(token.text)

        # Join with single space; we will run further normalization later
        return " ".join(out_tokens)

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
                        use_spacy_for_stopwords_and_names: bool = True,
                        ) -> str:
        """
        Main text preprocessing pipeline.

        Note: spaCy-based removals (stopwords & PERSON names) are applied BEFORE the regex-based masking
        so Named Entity Recognition sees raw tokens.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        text = self.normalize_unicode(text)

        # 1) spaCy-based removals (if enabled and available)
        if use_spacy_for_stopwords_and_names and self.use_spacy:
            text = self._apply_spacy_removals(text)

        # 2) Regex-based masking/removals
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

    def _unique_col_name(self, df: pd.DataFrame, base: str) -> str:
        """Return a column name not present in df by appending suffix numbers if needed."""
        if base not in df.columns:
            return base
        i = 1
        while f"{base}_{i}" in df.columns:
            i += 1
        return f"{base}_{i}"

    def preprocess_dataframe(self, df: pd.DataFrame,
                             columns: List[str],
                             keep_originals: bool = True,
                             cleaned_suffix: str = "_cleaned",
                             insert_cleaned_next_to_original: bool = True,
                             output_path: Optional[str] = None,
                             use_spacy_for_stopwords_and_names: bool = True,
                             **kwargs) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[int, str, str]]]]:
        """
        Apply preprocessing to each column in `columns`.
        - use_spacy_for_stopwords_and_names: whether to run spaCy-based removal inside preprocess_text.
        - Other flags and kwargs are forwarded to preprocess_text.
        Returns (processed_df, replacements_log).
        """
        df = df.copy()
        replacements_log: Dict[str, List[Tuple[int, str, str]]] = {}

        # Validate columns
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not in DataFrame")

        # Create cleaned columns
        created_clean_cols = []
        for col in columns:
            # build a desired name and ensure uniqueness
            desired_name = f"{col}{cleaned_suffix}" if keep_originals else col
            clean_col = self._unique_col_name(df, desired_name)
            created_clean_cols.append((col, clean_col))
            replacements_log[col] = []

            # initialize
            df[clean_col] = ""

            for idx, orig in df[col].items():
                try:
                    processed = self.preprocess_text(orig, use_spacy_for_stopwords_and_names=use_spacy_for_stopwords_and_names, **kwargs)
                except Exception:
                    # fallback to original string representation if preprocessing fails
                    processed = "" if orig is None else str(orig)
                df.at[idx, clean_col] = processed
                replacements_log[col].append((idx, orig, processed))

        # Reorder columns to place cleaned next to original (if requested)
        if insert_cleaned_next_to_original and keep_originals:
            orig_cols = list(df.columns)
            # We'll build a new ordered list
            new_order = []
            handled_clean_cols = {cc for _, cc in created_clean_cols}
            for c in orig_cols:
                # skip cleaned columns for now; we'll insert them after their originals
                if c in handled_clean_cols:
                    continue
                new_order.append(c)
                # if this original had a cleaned counterpart, insert it next
                for orig_col, clean_col in created_clean_cols:
                    if c == orig_col:
                        new_order.append(clean_col)
            # append any remaining columns (if any)
            for c in orig_cols:
                if c not in new_order:
                    new_order.append(c)
            df = df.reindex(columns=new_order)

        # Optionally save to CSV
        if output_path:
            df.to_csv(output_path, index=False)

        return df, replacements_log


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    demo = pd.DataFrame({
        "id": [1, 2, 3],
        "text": [
            "Email me at John.Doe@example.com about the meeting with Alice on 2023-08-12.",
            "Phone +91 98765 43210 and SSN 123-45-6789 at 10:34. Call Robert if needed.",
            "Credit card 4111 1111 1111 1111 and IP 192.168.0.1. Message to Samantha."
        ],
        "notes": [
            "Case# ABC-123 should remain. Contact: Priya.",
            None,
            "Follow-up: 2024/01/10 planned with team lead Mohan."
        ]
    })

    # You may need to install spaCy and the transformer model:
    # pip install -U "spacy[transformers]" && python -m spacy download en_core_web_trf
    #
    # If you don't want to install the heavy transformer model, the code will try to fall back to en_core_web_sm.

    tp = TextPreprocessor(
        spacy_model="en_core_web_trf",
        use_spacy=True,
        remove_spacy_stopwords=True,
        remove_persons=True
    )

    processed_df, log = tp.preprocess_dataframe(
        demo,
        columns=["text", "notes"],
        keep_originals=True,
        cleaned_suffix="_cleaned",
        insert_cleaned_next_to_original=True,
        output_path="processed_output.csv",
        lower=False,
        mask_emails=True,
        mask_phones=True,
        mask_urls=True,
        mask_ip=True,
        mask_credit=True,
        mask_ssn=True,
        remove_dates=True,
        remove_times=True,
        remove_punctuation=False,
        # control spaCy removal from client call as well:
        use_spacy_for_stopwords_and_names=True
    )

    # The CSV "processed_output.csv" will contain original columns plus the cleaned columns,
    # and each cleaned column will be placed right after its input column.
    print(processed_df)

Notes / tips

en_core_web_trf is large and requires spacy[transformers] plus model download; if you don't want the overhead, en_core_web_sm will work but may be less accurate for NER.

If performance is a concern for very large datasets, consider processing in batches or disabling spaCy (set use_spacy=False) and rely on lighter regex rules.

If you want to mask person names (replace with <PERSON>) instead of removing them, tell me and I’ll quickly switch the code to preserve token positions but replace PERSON spans with a token like <PERSON>.


Would you like me to change the behavior to mask person names (so you keep content shape) instead of deleting them?

