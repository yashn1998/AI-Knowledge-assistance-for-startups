Nice — you want a reusable, extensible text-preprocessing class that: (1) picks rows by Notetype, (2) strips email headers/greetings/quoted threads, (3) removes or masks PII using a mix of regex + spaCy NER, and (4) writes out a CSV with cleaned columns beside originals. Below is a production-ready, well-commented Python class that does exactly that and is easy to extend.

Two small notes before the code:

The script uses spaCy for NER and regex heuristics for things spaCy misses (emails, phones, urls, order numbers, headers). Install dependencies and the model if you haven't:

pip install pandas spacy regex

python -m spacy download en_core_web_sm


The heuristics aim to be conservative (so we don’t over-mask), and you can pass custom regexes or spaCy labels to control behavior.


Here’s the class. Copy-paste and run.

"""
TextPreprocessor
- Accepts a pandas DataFrame
- Filters rows by Notetype values
- Cleans email headers, greetings, quoted threads
- Removes or masks PII using regex + spaCy NER
- Outputs CSV with cleaned columns appended as "<col>_cleaned"
- Easy to extend: add more regex patterns or NER labels
"""

from typing import List, Optional, Dict, Tuple
import re
import regex  # pip install regex (better unicode handling)
import pandas as pd
import spacy
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextPreprocessor:
    nlp_model: str = "en_core_web_sm"
    mask_token_format: str = "<{TYPE}>"   # e.g. "<EMAIL>" or "<PERSON>"
    mask: bool = True                     # True=>replace PII with mask token, False=>delete PII
    verbose: bool = True
    custom_regexes: Dict[str, str] = field(default_factory=dict)  # name -> pattern
    custom_entity_labels_to_mask: Optional[List[str]] = None      # spaCy labels to mask (default below)
    remove_quoted_threads: bool = True
    remove_headers: bool = True
    remove_greetings_signoffs: bool = True

    def __post_init__(self):
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        # Load spaCy model lazily to defer heavy import if not used
        try:
            self.nlp = spacy.load(self.nlp_model)
        except Exception as e:
            logger.error(f"Error loading spaCy model {self.nlp_model}: {e}")
            logger.error("Make sure you installed the model: python -m spacy download en_core_web_sm")
            raise

        # Default spaCy labels we consider PII-ish unless user overrides
        if self.custom_entity_labels_to_mask is None:
            self.entity_labels = {
                "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "NORP", "FAC"
            }
        else:
            self.entity_labels = set(self.custom_entity_labels_to_mask)

        # sensible default regex-based PII detectors
        self.regex_patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "URL": r"https?://\S+|www\.\S+",
            "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "PHONE": r"(\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{6,12}\b",
            # common ticket/order identifiers used in enterprise notes (adjust if needed)
            "SR_ORDER": r"\b(?:SR|Service Request|ORDER(?: NUMBER)?|Order ID)[:#]?\s*[A-Za-z0-9\-_/]+\b",
            # simple credit card-ish detection (not perfect)
            "CARD": r"\b(?:\d[ -]*?){13,19}\b",
        }
        # allow user-supplied overrides/additions
        self.regex_patterns.update(self.custom_regexes)

        # patterns for headers/greetings/quoted email blocks
        self.header_keys = [
            r"^From:", r"^Sent:", r"^To:", r"^Cc:", r"^Bcc:", r"^Subject:", r"^Date:", r"^Order Creation",
            r"^Order Number", r"^ROUTING DECISION", r"^COLLABORATION ACTIVITY", r"^ATTACHMENT ADDED",
            r"^AUTOMATION LOG", r"^CURRENT STATUS", r"^RESOLUTION SUMMARY"
        ]
        # greetings/signoffs (a conservative list)
        self.greetings_pattern = r"^(hi|hello|dear|greetings|team|hey)[\s,!:].{0,80}"
        self.signoff_starts = [
            r"^regards", r"^thanks", r"^thank you", r"^best regards", r"^cheers", r"^sincerely", r"^br,"
        ]

        # compiled regex for efficiency
        self._compiled = {k: re.compile(v, flags=re.IGNORECASE) for k, v in self.regex_patterns.items()}
        self._greet_re = re.compile(self.greetings_pattern, flags=re.IGNORECASE | re.MULTILINE)
        self._signoff_re = re.compile("|".join(self.signoff_starts), flags=re.IGNORECASE | re.MULTILINE)

        # quoted block markers
        self.quote_markers = [
            r"-----Original Message-----", r"On .* wrote:", r"From: .*?To: .*?Subject:",
            r"____________________________", r"--- Forwarded message ---"
        ]
        self._quote_re = re.compile("|".join([re.escape(m) for m in self.quote_markers]), flags=re.IGNORECASE | re.DOTALL)

    def preprocess_df(self,
                      df: pd.DataFrame,
                      notetype_values: List[str],
                      text_columns: List[str],
                      output_csv_path: Optional[str] = None,
                      mask: Optional[bool] = None) -> pd.DataFrame:
        """
        Main entry:
         - df: input dataframe
         - notetype_values: list of Notetype values to process (rows filtered on column 'Notetype' - case-insensitive)
         - text_columns: list of column names to clean
         - output_csv_path: if provided, save resulting df to this CSV
         - mask: override instance mask
        Returns new dataframe (copy) with new columns "<col>_cleaned"
        """
        if mask is None:
            mask = self.mask

        if "Notetype" not in df.columns:
            raise ValueError("DataFrame must contain a 'Notetype' column.")

        df_out = df.copy()
        # boolean mask for rows to process (case insensitive match)
        mask_rows = df_out["Notetype"].astype(str).str.lower().isin([v.lower() for v in notetype_values])

        logger.info(f"Rows matching Notetype values: {mask_rows.sum()} / {len(df_out)}")

        for col in text_columns:
            if col not in df_out.columns:
                raise ValueError(f"Text column '{col}' not found in dataframe.")
            out_col = f"{col}_cleaned"
            # default: copy original
            df_out[out_col] = df_out[col].astype(str)

            # apply row-wise on selected rows
            for idx in df_out[mask_rows].index:
                original = df_out.at[idx, col]
                cleaned = self._clean_text(str(original), mask=mask)
                df_out.at[idx, out_col] = cleaned

        # Optionally save CSV
        if output_csv_path:
            df_out.to_csv(output_csv_path, index=False)
            logger.info(f"Wrote cleaned dataframe to {output_csv_path}")

        return df_out

    # ---------- Core cleaning pipeline ----------
    def _clean_text(self, text: str, mask: bool = True) -> str:
        working = text

        # 1) Normalize line endings
        working = working.replace("\r\n", "\n").replace("\r", "\n")

        # 2) Remove leading/trailing whitespace
        working = working.strip()

        # 3) Remove email headers (top blocks that look like header fields)
        if self.remove_headers:
            working = self._remove_email_headers(working)

        # 4) Remove quoted threads (everything from original message markers onward)
        if self.remove_quoted_threads:
            working = self._remove_quoted_thread(working)

        # 5) Remove greetings / signoffs
        if self.remove_greetings_signoffs:
            working = self._remove_greetings_and_signoffs(working)

        # 6) PII detection & masking/removal (regex + spaCy)
        working = self._mask_or_remove_pii(working, mask=mask)

        # 7) Normalize whitespace: collapse multiple spaces and >2 newlines to single newline
        working = re.sub(r"[ \t]{2,}", " ", working)
        working = re.sub(r"\n{3,}", "\n\n", working)
        working = working.strip()

        return working

    # ---------- header & thread removal helpers ----------
    def _remove_email_headers(self, text: str) -> str:
        """
        Heuristic: remove contiguous header-like lines at the top.
        Also remove any lines that start with known header keys anywhere.
        """
        lines = text.split("\n")
        cleaned_lines = []
        header_mode = True
        for i, line in enumerate(lines):
            stripped = line.strip()
            if header_mode:
                # if line empty -> might be end of header
                if stripped == "":
                    # once blank line occurs, exit header mode
                    header_mode = False
                    continue
                # if line starts with known header key then skip it
                starts_header = any(re.match(k, stripped, flags=re.IGNORECASE) for k in self.header_keys)
                # also skip long comma-separated From/To lines which often contain emails
                if starts_header or re.search(r"\bFrom:.*\bTo:|\bTo:.*\bFrom:", stripped, flags=re.IGNORECASE):
                    continue
                # if line looks like "ORDER NUMBER: ..." skip
                if re.match(r"^(ORDER|Order|ROUTING|COLLABORATION|ATTACHMENT|AUTOMATION|CURRENT STATUS|RESOLUTION)", stripped, flags=re.IGNORECASE):
                    continue
                # if line contains many emails (more than 1) drop it
                if len(re.findall(self._compiled["EMAIL"], stripped)) >= 1 and len(stripped) < 400:
                    # header line with emails -> drop
                    continue
                # if no clear break and the line is short but contains ":" fields like "Subject: ..." skip
                if ":" in stripped and len(stripped) < 200 and re.match(r"^[A-Za-z\s]+:", stripped):
                    continue
                # once we see something that doesn't look like a header, switch out of header_mode
                header_mode = False
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _remove_quoted_thread(self, text: str) -> str:
        # remove from common quoted markers to the end
        m = self._quote_re.search(text)
        if m:
            return text[:m.start()].strip()
        return text

    def _remove_greetings_and_signoffs(self, text: str) -> str:
        # remove common greeting at start (one line)
        text = re.sub(r"^\s*" + self.greetings_pattern + r"(?:\r?\n)?", "", text, flags=re.IGNORECASE | re.MULTILINE)
        # remove signoff lines near the end
        # find last 6 lines and drop signoff-starting line to end
        lines = text.split("\n")
        L = len(lines)
        for i in range(max(0, L-6), L):
            if self._signoff_re.match(lines[i].strip()):
                # chop from this signoff to the end
                return "\n".join(lines[:i]).strip()
        return text

    # ---------- PII detection & replacement ----------
    def _mask_or_remove_pii(self, text: str, mask: bool = True) -> str:
        """
        Build list of spans from regex + spaCy, then replace from end->start to maintain offsets.
        Replacement: either mask token "<TYPE>" or empty string (delete).
        """
        spans: List[Tuple[int,int,str]] = []

        # 1) regex patterns
        for ptype, cre in self._compiled.items():
            for m in cre.finditer(text):
                s, e = m.span()
                spans.append((s, e, ptype.upper()))

        # 2) spaCy entities
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in self.entity_labels:
                    spans.append((ent.start_char, ent.end_char, ent.label_))
        except Exception as e:
            # spaCy might fail on some long texts; log and continue
            logger.debug(f"spaCy failed: {e}")

        if not spans:
            return text

        # merge overlapping spans and keep the largest span per overlap
        spans = sorted(spans, key=lambda x: (x[0], -x[1]))
        merged: List[Tuple[int,int,str]] = []
        for s,e,t in spans:
            if not merged:
                merged.append((s,e,t))
            else:
                last_s, last_e, last_t = merged[-1]
                if s <= last_e:  # overlap
                    # extend end if needed
                    if e > last_e:
                        # create new merged
                        merged[-1] = (last_s, e, f"{last_t}|{t}")
                    else:
                        # keep existing
                        pass
                else:
                    merged.append((s,e,t))
        # perform replacements from end to start
        out = text
        for s,e,t in reversed(merged):
            if mask:
                token = self.mask_token_format.format(TYPE=t.split("|")[0])
                out = out[:s] + token + out[e:]
            else:
                out = out[:s] + out[e:]
        return out

# ---------------- Example usage ----------------

if __name__ == "__main__":
    # Example:
    import io
    data = {
        "Notetype": ["Email In", "Other"],
        "NOTE": [
            "From: alice@example.com To: bob@cisco.com Subject: Foo\nHi team,\nPlease see ORDER NUMBER: 804254450 CREATED DATE: 25-Aug-2025\nRegards,\nAlice\n-----Original Message-----\nFrom: someone else",
            "This row should stay untouched"
        ]
    }
    df = pd.DataFrame(data)
    tp = TextPreprocessor(mask_token_format="<PII>", mask=True, verbose=True)
    cleaned = tp.preprocess_df(df, notetype_values=["Email In"], text_columns=["NOTE"], output_csv_path=None)
    print(cleaned[["NOTE", "NOTE_cleaned"]].to_string(index=False))

How it addresses your requirements

Loads a DataFrame and processes rows selected by Notetype.

Removes email headers (heuristic lines like From:, To:, Subject:, ORDER NUMBER, etc.) and quoted thread sections (-----Original Message-----, On ... wrote:).

Removes greetings and signoffs conservatively.

Detects and masks/removes PII using:

regex for emails, URLs, IPs, phones, SR/order IDs, cards;

spaCy NER for PERSON, ORG, GPE, DATE, etc. (configurable).


Option to mask (default) or delete PII (pass mask=False).

Produces cleaned columns as COLUMN_cleaned and can save CSV.


How to extend

Add more regexes to custom_regexes when constructing TextPreprocessor.

Add or remove spaCy labels via custom_entity_labels_to_mask.

Add specialized header removal rules in _remove_email_headers.

Hook in additional steps (spell-correction, tokenization, normalization) by adding methods and calling them in _clean_text.


That should get you a robust starting point for LLM-prep text cleaning. If you want, I can:

Add support for batching/parallel processing for large DataFrames,

Add a summary log (counts of masked emails/phones/etc.),

Or tune header-removal to your exact screenshot patterns (I can extract stricter header keywords if you paste a few raw examples).


