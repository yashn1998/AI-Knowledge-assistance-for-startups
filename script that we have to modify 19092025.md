"""
Generic, robust text preprocessing for Pandas DataFrames.

Features:
- Apply to any dataframe columns you specify.
- Remove/mask: URLs, emails, phones, credit cards, IPs, SSNs, timestamps/dates, addresses,
  generic PII (PERSON/ORG/LOC) via spaCy (optional).
- Keep or remove special "case id / serial" patterns (configurable).
- Remove punctuation, non-alpha noise (configurable).
- Returns processed DataFrame and optional replacements log.

Dependencies: pandas, regex, unicodedata. spaCy optional for NER-based PII removal.

Example:
    pre = TextPreprocessor()
    df2, log = pre.preprocess_dataframe(df, cols=['desc','notes'],
                                        keep_case_ids=True,
                                        mask_emails=True,
                                        remove_dates=True,
                                        use_spacy=False)
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Optional
import pandas as pd

# Optional spaCy import deferred to runtime if user requests NER-based removal
_SPACY_AVAILABLE = False
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False


class TextPreprocessor:
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        self.nlp = None
        self.nlp_model = nlp_model

        # Regex patterns:
        self.re_url = re.compile(r"""(?i)\b((?:https?://|www\.)\S+)\b""")
        self.re_email = re.compile(r"""([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})""")
        # phone numbers: international and local common patterns
        self.re_phone = re.compile(r"""(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?[\d\-.\s]{6,15}\d""")
        # IP
        self.re_ip = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        # credit card like sequences (very generic)
        self.re_credit = re.compile(r'\b(?:\d[ -]*?){13,19}\b')
        # SSN (US common)
        self.re_ssn = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        # timestamps / datetime (many formats)
        self.re_datetime = re.compile(
            r'\b(?:\d{1,2}[:/-]\d{1,2}[:/-]\d{2,4}|\d{4}[-/]\d{2}[-/]\d{2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',
            re.IGNORECASE)
        # generic timestamp with time
        self.re_time = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APMapm]{2})?\b')
        # url/links clean (with scheme or not)
        self.re_url2 = re.compile(r'\bhttps?://\S+|\bwww\.\S+\b', re.IGNORECASE)
        # addresses / zip-like patterns (basic heuristics)
        self.re_zip = re.compile(r'\b\d{5}(?:-\d{4})?\b')
        # serial/case id detection: uppercase letters/digits groups separated by hyphens (configurable)
        self.re_case_id = re.compile(r'\b[A-Z]{2,}\d{0,3}(?:-[A-Z0-9]+){2,}\b')
        # remove punctuation but keep hyphens if asked
        self.re_punct = re.compile(r'[^\w\s-]')
        # non-ascii/ non-printable characters
        self.re_nonprint = re.compile(r'[\x00-\x1f\x7f-\x9f]')
        # long dash lines / all punctuation sequences
        self.re_punctseq = re.compile(r'^[\W_]+$')

        # Words to preserve as tokens if masked
        self._mask_tokens = {
            'email': '<EMAIL>',
            'phone': '<PHONE>',
            'url': '<URL>',
            'ip': '<IP>',
            'date': '<DATE>',
            'time': '<TIME>',
            'credit': '<CREDIT>',
            'ssn': '<SSN>',
            'address': '<ADDRESS>',
            'case_id': '<CASE_ID>',
            'person': '<PERSON>',
            'org': '<ORG>',
            'location': '<LOCATION>',
        }

    def _ensure_spacy(self):
        if self.nlp is None:
            if not _SPACY_AVAILABLE:
                raise RuntimeError("spaCy not available. Install spaCy and a model (eg. en_core_web_sm) to use NER features.")
            try:
                self.nlp = spacy.load(self.nlp_model)
            except Exception:
                # try download if missing — user can add their own installation step
                raise RuntimeError(f"spaCy model {self.nlp_model} not found. Install it: python -m spacy download {self.nlp_model}")

    def normalize_unicode(self, text: str) -> str:
        # NFC normalization and remove non-printables
        text = unicodedata.normalize("NFC", text)
        text = self.re_nonprint.sub(" ", text)
        return text

    def mask_with_placeholder(self, text: str, pattern: re.Pattern, placeholder: str, mask: bool=True) -> str:
        if mask:
            return pattern.sub(placeholder, text)
        else:
            # default remove
            return pattern.sub("", text)

    def remove_addresses_heuristic(self, text: str) -> str:
        # Basic heuristics: remove common address patterns like '1234 Street Name, City, ST 12345'
        # This is intentionally conservative.
        # Look for patterns: number + street word + city/state + zip
        addr_pattern = re.compile(r'\b\d{1,5}\s+[A-Za-z0-9\.\-]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Drive|Dr|Court|Ct|Way|Place|Pl)\b.*?(?:,?\s*[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5})?', re.IGNORECASE)
        text = addr_pattern.sub(self._mask_tokens['address'], text)
        # catch zip lines
        text = self.re_zip.sub(self._mask_tokens['address'], text)
        return text

    def remove_persons_spacy(self, text: str) -> str:
        # Remove PERSON/ORG/GPE etc using spaCy, replace with placeholders
        if not _SPACY_AVAILABLE:
            return text
        if self.nlp is None:
            self._ensure_spacy()
        doc = self.nlp(text)
        out = []
        last = 0
        for ent in doc.ents:
            # only care about PERSON, ORG, GPE, LOC, FAC
            if ent.label_ in ('PERSON', 'ORG', 'GPE', 'LOC', 'FAC'):
                label = ent.label_.lower()
                mask = self._mask_tokens.get('person') if ent.label_ == 'PERSON' else self._mask_tokens.get('org') if ent.label_ == 'ORG' else self._mask_tokens.get('location')
                # splice text
                out.append(text[last:ent.start_char])
                out.append(mask)
                last = ent.end_char
        out.append(text[last:])
        return "".join(out)

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
                        remove_non_alpha_tokens: bool = False,
                        keep_case_ids: bool = True,
                        mask_case_ids: bool = False,
                        remove_addresses: bool = True,
                        use_spacy_ner: bool = False
                        ) -> str:
        """
        Process a single text string according to options.
        """
        if text is None:
            return text
        if not isinstance(text, str):
            text = str(text)

        # normalize
        text = self.normalize_unicode(text)

        # optionally preserve case IDs by temporarily marking them
        case_id_tokens = {}
        if keep_case_ids:
            def _case_repl(m):
                tok = f"__CASEID_{len(case_id_tokens)}__"
                case_id_tokens[tok] = m.group(0)
                return tok
            text = self.re_case_id.sub(_case_repl, text)
        elif mask_case_ids:
            text = self.re_case_id.sub(self._mask_tokens['case_id'], text)

        # URLs
        if mask_urls:
            text = self.re_url2.sub(self._mask_tokens['url'], text)

        # emails
        if mask_emails:
            text = self.re_email.sub(self._mask_tokens['email'], text)

        # phones
        if mask_phones:
            text = self.re_phone.sub(self._mask_tokens['phone'], text)

        # IPs
        if mask_ip:
            text = self.re_ip.sub(self._mask_tokens['ip'], text)

        # credit-like sequences
        if mask_credit:
            text = self.re_credit.sub(self._mask_tokens['credit'], text)

        # SSN
        if mask_ssn:
            text = self.re_ssn.sub(self._mask_tokens['ssn'], text)

        # dates / times
        if remove_dates:
            text = self.re_datetime.sub(self._mask_tokens['date'], text)
        if remove_times:
            text = self.re_time.sub(self._mask_tokens['time'], text)

        # addresses heuristic
        if remove_addresses:
            text = self.remove_addresses_heuristic(text)

        # NER-based PERSON/ORG/LOC removal (optional, requires spaCy)
        if use_spacy_ner:
            try:
                text = self.remove_persons_spacy(text)
            except RuntimeError:
                # spaCy missing or model missing -> ignore but warn
                pass

        # Remove punctuation if asked (keep hyphens and underscores if desired)
        if remove_punctuation:
            text = self.re_punct.sub(" ", text)

        # Remove tokens that are pure punctuation or long sequences
        text = re.sub(r'\s{2,}', ' ', text).strip()

        # remove tokens not English-like (very conservative): tokens with >50% non-alpha
        if remove_non_alpha_tokens:
            toks = text.split()
            filtered = []
            for t in toks:
                alpha_chars = sum(1 for ch in t if ch.isalpha())
                if len(t) == 0:
                    continue
                if alpha_chars / max(1, len(t)) < 0.5:
                    # skip numeric-heavy tokens except preserved placeholders or case ids
                    if t in case_id_tokens or t in self._mask_tokens.values():
                        filtered.append(t)
                    else:
                        # drop token
                        continue
                else:
                    filtered.append(t)
            text = " ".join(filtered)

        # restore preserved case ids
        if case_id_tokens:
            for placeholder, orig in case_id_tokens.items():
                text = text.replace(placeholder, orig)

        # final cleanup
        text = re.sub(r'\s{2,}', ' ', text).strip()
        if lower:
            text = text.lower()

        return text

    def preprocess_dataframe(self, df: pd.DataFrame,
                             columns: List[str],
                             inplace: bool = False,
                             **kwargs) -> Tuple[pd.DataFrame, Optional[Dict[str, List[Tuple[int,str,str]]]]]:
        """
        Apply preprocess_text to all specified columns.

        Returns:
            processed_df: DataFrame copy (or same if inplace True)
            log (optional): dictionary mapping column -> list of (index, original, processed) entries
        Options (**kwargs) are forwarded to preprocess_text.
        """
        if not inplace:
            df = df.copy()

        replacements_log = {col: [] for col in columns}

        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not in DataFrame")
            for idx, orig in df[col].items():
                try:
                    processed = self.preprocess_text(orig, **kwargs)
                except Exception as e:
                    # on error, keep original and log
                    processed = orig
                df.at[idx, col] = processed
                # small log for debugging / auditing
                replacements_log[col].append((int(idx) if hasattr(idx, "__int__") else idx, orig, processed))

        return df, replacements_log


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Quick demo DataFrame to test many cases (you can adapt with your real data)
    demo = pd.DataFrame({
        "id": [1, 2, 3],
        "text": [
            "Problem Description is clean. Serial: FRA21-0101-0400-05T2. Contact: Michael.Brawner@microsoft.com +1 (703) 234-2526. 41840 Growth Mindset Lane, Aldie, VA 20105. Requested Delivery Date: August 18.",
            "Overlapping data, dashes --- timestamps 2023-07-15 10:34:22 and url http://example.com. Case# CASE-2024-ABC-002",
            "Address: 1234 Elm St, Smalltown, CA 90210. Email: someone@company.org. IP: 192.168.0.1. CC: 4111 1111 1111 1111"
        ]
    })

    tp = TextPreprocessor()

    processed_df, log = tp.preprocess_dataframe(
        demo,
        columns=["text"],
        inplace=False,
        lower=False,  # preserve case if you want to keep meaning for case ids
        mask_emails=True,
        mask_phones=True,
        mask_urls=True,
        mask_ip=True,
        mask_credit=True,
        mask_ssn=True,
        remove_dates=True,
        remove_times=True,
        remove_punctuation=False,
        remove_non_alpha_tokens=False,
        keep_case_ids=True,
        mask_case_ids=False,
        remove_addresses=True,
        use_spacy_ner=False  # set True if spaCy installed and you want PERSON/ORG removed
    )

    print(processed_df.to_dict(orient='records'))
    # Optionally inspect log
    # print(log)



keep above script  first as level 1 cleaning of data 


below things we have to modify and do improvements in it and make it as level 2 clean 


1. in this we want output in CSV file . 
2. Currently this is removing any number from and remove  .so fix this as well . 
3. use spacy advance model 
and do lematizationn
4. we have to remove stopwords also also give option to add custom stopwords . 
5. 

