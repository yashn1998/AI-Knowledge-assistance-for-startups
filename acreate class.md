Nice — I read the screenshots and translated the instructions into a robust TextPreprocessor you can drop into your project. This implementation:

Removes email headers at the top while preserving the Subject line.

Strips quoted threads (common markers like -----Original Message-----, On ... wrote:, > quote).

Removes greetings and signoffs heuristically (e.g., "Hi", "Regards", "Sent from my …").

Masks or removes PII (email, phone, URLs, IPs, and common IDs). mask=True inserts tokens like <EMAIL>; mask=False removes them entirely.

Removes repeated punctuation/line separators (e.g., -----, ______, ***) if length ≥ 3.

Removes filenames (common extensions) and supports removal of custom keywords.

Normalizes whitespace and trims leading/trailing spaces.


Drop it in and call _clean_text(text, mask=True) to mask PII or _clean_text(text, mask=False) to remove them. I included optional spaCy support (if installed) to improve PII detection — it falls back to regex when spaCy isn't available.

Here’s the code:

import re
from typing import List, Optional

try:
    import spacy
    _HAS_SPACY = True
    _NLP = None
except Exception:
    _HAS_SPACY = False
    _NLP = None


class TextPreprocessor:
    """
    Text preprocessor for email/notes cleaning:
     - normalize line endings
     - remove top email headers (keeps Subject:)
     - remove quoted threads (Original Message / On ... wrote:)
     - remove greetings and signoffs
     - mask or remove PII (email, phone, url, ip, ids)
     - remove repeated separators (---, ___, ***)
     - remove filenames and custom keywords
     - normalize whitespace
    """

    def __init__(
        self,
        custom_keywords: Optional[List[str]] = None,
        remove_filenames: bool = True,
        min_separator_len: int = 3
    ):
        self.custom_keywords = custom_keywords or []
        self.remove_filenames = remove_filenames
        self.min_separator_len = min_separator_len

        global _NLP
        if _HAS_SPACY and _NLP is None:
            try:
                _NLP = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
            except Exception:
                _NLP = None  # spaCy exists but model not installed

    # ---------------- core pipeline ----------------
    def _clean_text(self, text: str, mask: bool = True) -> str:
        """
        Central cleaning pipeline. `mask=True` will replace PII with tokens like <EMAIL>.
        `mask=False` will remove PII entirely.
        """
        if not text:
            return ""

        # 1) Normalize line endings
        working = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2) Remove leading/trailing whitespace
        working = working.strip()

        # 3) Remove email headers (top blocks that look like header fields)
        working = self._remove_email_headers(working)

        # 4) Remove quoted threads (everything from common quoted markers onward)
        working = self._remove_quoted_thread(working)

        # 5) Remove greetings / signoffs
        working = self._remove_greetings_and_signoffs(working)

        # 6) PII detection & masking/removal (regex + optional spaCy)
        working = self._mask_or_remove_pii(working, mask=mask)

        # 7) Remove file names if requested
        if self.remove_filenames:
            working = self._remove_filenames(working)

        # 8) Remove custom keywords (whole-word)
        for kw in self.custom_keywords:
            working = re.sub(rf"\b{re.escape(kw)}\b", "", working, flags=re.IGNORECASE)

        # 9) Remove long repeated separators/lines like '-----', '______', '***'
        working = self._remove_repeated_separators(working, min_len=self.min_separator_len)

        # 10) Normalize whitespace: collapse multiple spaces, collapse >2 newlines to single newline
        working = re.sub(r"[ \t]{2,}", " ", working)             # collapse many spaces/tabs
        working = re.sub(r"\n{2,}", "\n\n", working)              # keep at most 1 blank line between paragraphs
        working = working.strip()

        return working

    # ---------------- helper functions ----------------
    def _remove_email_headers(self, text: str) -> str:
        """
        Remove header-like lines at the start of the text, but preserve the Subject: line.
        Header block definition: contiguous lines from beginning that match a header key pattern,
        ending at first blank line or the first non-header line.
        """
        HEADER_KEYS = (
            r"From|To|Cc|Bcc|Date|Sent|Received|Reply-To|Return-Path|Message-ID|In-Reply-To|"
            r"References|X-|MIME-Version|Content-Type|Content-Transfer-Encoding|Delivered-To|"
            r"E-?mail|Phone|Tel|Fax"
        )
        header_re = re.compile(rf"^\s*({HEADER_KEYS})\s*:", flags=re.IGNORECASE)
        subject_re = re.compile(r"^\s*Subject\s*:\s*(.*)$", flags=re.IGNORECASE)

        lines = text.split("\n")
        if not lines:
            return text

        # If the very first non-empty line is not a header we probably don't have a header block.
        # We'll scan from the top for a contiguous header-like block.
        keep_lines = []
        in_header_block = False
        header_block_done = False
        for idx, line in enumerate(lines):
            if idx == 0 and line.strip() == "":
                # leading empty lines — keep and continue
                keep_lines.append(line)
                continue

            if header_re.match(line):
                in_header_block = True
                # If Subject line, preserve it
                subj_m = subject_re.match(line)
                if subj_m:
                    # Keep Subject line exactly as is (trim trailing)
                    keep_lines.append("Subject: " + subj_m.group(1).strip())
                # else — drop other header lines
                continue
            else:
                if in_header_block:
                    # first non-header after header block -> header block ends. Keep this and following.
                    in_header_block = False
                    header_block_done = True
                    keep_lines.append(line)
                else:
                    # no header block started: just keep the rest of the text
                    keep_lines.append(line)

            # if header block already done, just append remaining lines
            if header_block_done:
                keep_lines.extend(lines[idx + 1:])
                break

        # If everything was header-like and we didn't append anything else, ensure we keep any Subject captured.
        if not keep_lines:
            # fallback just return original (safe)
            return text

        # join and return
        result = "\n".join(keep_lines)
        # remove any leading/trailing blank lines
        return result.strip()

    def _remove_quoted_thread(self, text: str) -> str:
        """
        Remove everything from common quoted-thread markers onward.
        Markers include:
          - -----Original Message-----
          - On <date>, <name> wrote:
          - > (quoted lines)
          - ----- Forwarded message -----
        """
        markers = [
            r"^-{2,}\s*Original Message\s*-{2,}",      # -----Original Message-----
            r"^-{2,}\s*Forwarded message\s*-{2,}",     # ----- Forwarded message -----
            r"^On .* wrote:",                          # On Jan 1, 2020, John Doe <...> wrote:
            r"^From: .*",                              # From: someone <...>
            r"^\>+",                                   # lines starting with >
            r"^-----\s*Original\s*-----",              # other variants
            r"^Begin forwarded message:",               # Outlook/others
        ]
        combined = re.compile("|".join(markers), flags=re.IGNORECASE | re.MULTILINE)
        m = combined.search(text)
        if m:
            # keep only the part before the first marker
            return text[: m.start()].strip()
        return text

    def _remove_greetings_and_signoffs(self, text: str) -> str:
        """
        Remove salutations at the start and signoffs at the end.
        - greeting patterns at start like "Hi John,", "Dear Team," -> remove first 1-2 lines
        - signoff patterns like "Regards," "Thanks," "Best," -> remove from that signoff line to end (remove signature)
        """
        lines = text.split("\n")

        # Remove a top greeting if present (first few lines)
        if len(lines) > 0:
            top_sample = " ".join(lines[:2]).lower()
            if re.match(r"^\s*(hi|hello|dear|greetings|hey)\b", lines[0], flags=re.IGNORECASE) or \
               re.match(r"^\s*(hi|hello|dear|greetings|hey)\b", top_sample, flags=re.IGNORECASE):
                # drop first line (greeting)
                lines = lines[1:]
                # if next line is short name/empty, drop again
                if lines and re.match(r"^\s*[A-Z][a-z]+\s*$", lines[0]):
                    lines = lines[1:]

        # Remove signoff block: find signoff keywords and drop everything from that line to end
        signoff_keys = [
            r"regards\b", r"best regards\b", r"best\b", r"thanks\b", r"thank you\b", r"sincerely\b",
            r"cheers\b", r"warm regards\b", r"yours sincerely\b", r"sent from my", r"kind regards\b",
            r"thanks,\s*team\b", r"thank you,\s*team\b"
        ]
        signoff_re = re.compile(rf"^\s*({'|'.join(signoff_keys)})", flags=re.IGNORECASE)
        for i, ln in enumerate(lines):
            if signoff_re.match(ln):
                # drop from i to end
                lines = lines[:i]
                break

        # Also, if a block at end contains lots of contact info (phone/email lines), drop last upto 6 lines
        tail = lines[-6:] if len(lines) >= 6 else lines
        contact_hint = re.compile(r"(tel|phone|fax|mobile|@|mailto:|www\.|http)", flags=re.IGNORECASE)
        # if majority of tail lines contain contact hint, remove them
        if tail and sum(1 for l in tail if contact_hint.search(l)) >= max(1, len(tail) // 2):
            # remove trailing lines that include contact hints
            new_len = len(lines)
            for j in range(len(lines) - 1, -1, -1):
                if contact_hint.search(lines[j]):
                    new_len = j
                else:
                    # stop when we hit a line that doesn't look like contact
                    break
            lines = lines[:new_len]

        return "\n".join(lines).strip()

    def _mask_or_remove_pii(self, text: str, mask: bool = True) -> str:
        """
        Mask or remove PII using regex. If spaCy NER is available it will also be used to
        detect PERSON/ORG/LOC/GPE/DATE optionally (can be extended).
        Mask tokens used: <EMAIL>, <PHONE>, <URL>, <IP>, <NUMBER>
        """
        # Regex patterns
        # Email
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<EMAIL>" if mask else "", text)

        # mailto: style
        text = re.sub(r"mailto:\s*\S+", "<EMAIL>" if mask else "", text, flags=re.IGNORECASE)

        # URLs
        text = re.sub(r"\bhttps?://\S+\b", "<URL>" if mask else "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwww\.\S+\b", "<URL>" if mask else "", text, flags=re.IGNORECASE)

        # Phone numbers: common international +1-234-567-8900, (123) 456-7890, 123.456.7890 etc.
        phone_pattern = re.compile(
            r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?[\d\-.\s]{6,15}\d"
        )
        text = phone_pattern.sub("<PHONE>" if mask else "", text)

        # IP addresses
        text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>" if mask else "", text)

        # Numbers / IDs - be cautious: we only remove long numeric tokens that look like IDs
        text = re.sub(r"\b[A-Z0-9]{6,}\b", "<ID>" if mask else "", text)  # alphanumeric long IDs
        text = re.sub(r"\b\d{6,}\b", "<NUMBER>" if mask else "", text)

        # If spaCy is available, run NER pass to mask PERSON/ORG/LOC etc.
        if _NLP:
            try:
                doc = _NLP(text)
                # we'll replace named entities with tokens (PERSON -> <PERSON>, etc.)
                ents = []
                for ent in doc.ents:
                    if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "CARDINAL"}:
                        ents.append((ent.start_char, ent.end_char, ent.label_))
                # replace from end to start to avoid index shifts
                if ents:
                    new_text = []
                    last = 0
                    for s, e, label in sorted(ents, key=lambda x: x[0]):
                        new_text.append(text[last:s])
                        token = f"<{label}>"
                        new_text.append(token if mask else "")
                        last = e
                    new_text.append(text[last:])
                    text = "".join(new_text)
            except Exception:
                pass  # if spaCy fails, we fallback to the regex substitutions above

        # collapse multiple adjacent tokens like "<EMAIL><PHONE>" with a space
        text = re.sub(r"(>(\s*<))+", "> <", text)

        # finally strip repeated or stray punctuation leftover
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _remove_filenames(self, text: str) -> str:
        """
        Remove filenames with common file extensions.
        """
        file_exts = r"pdf|docx|doc|xls|xlsx|csv|txt|ppt|pptx|zip|tar|gz|png|jpg|jpeg|bmp|svg"
        text = re.sub(rf"\b[\w\-\s]+\.({file_exts})\b", "", text, flags=re.IGNORECASE)
        # also remove inline paths like C:\Users\... or /home/.../file.ext
        text = re.sub(r"(?:[A-Za-z]:\\|/)[\w\\/\-\. ]+\.\w{2,6}", "", text)
        return text

    def _remove_repeated_separators(self, text: str, min_len: int = 3) -> str:
        """
        Remove lines that are separators of the same character repeated min_len or more times,
        like '-----', '___', '***'.
        """
        def _is_sep_line(s: str) -> bool:
            s_stripped = s.strip()
            if len(s_stripped) < min_len:
                return False
            # only punctuation characters allowed
            if all(ch in "-_=*#~.·•" for ch in s_stripped):
                # check majority same char
                return True
            # underscores and hyphens mixed count too, but require length
            if len(re.sub(r"[\-_=*#~.·•]+", "", s_stripped)) == 0 and len(s_stripped) >= min_len:
                return True
            return False

        lines = [ln for ln in text.split("\n") if not _is_sep_line(ln)]
        return "\n".join(lines).strip()


# ---------------- Example usage ----------------
if __name__ == "__main__":
    tp = TextPreprocessor(custom_keywords=["CONFIDENTIAL", "DO_NOT_USE"])
    raw = """From: John Doe <john@example.com>
Date: Tue, 1 Jan 2024 12:00:00 +0000
Subject: Meeting notes about RMA issues
To: team@example.com

Hi team,

Please see attached file notes.pdf and let me know.

-----Original Message-----
From: Jane <jane@example.com>
On Jan 1, 2024, Jane wrote:
> previous message

Regards,
John
Phone: +1 919-392-3805
"""
    cleaned = tp._clean_text(raw, mask=True)
    print("CLEANED:\n", cleaned)


---

Notes & suggestions (practical):

If you want stricter header parsing (e.g., preserve other fields or capture Subject separately), we can extract Subject: into a metadata dict and return (clean_text, metadata).

If you prefer no masking tokens at all (to save tokens in LLM input), call _clean_text(..., mask=False).

SpaCy improves detection of people/orgs but is optional; the regex is the core functionality and will work without it.

If the current heuristics remove something they shouldn't, give me a couple of real small examples and I’ll tighten the rules (or make exceptions) — I’ll bias towards preserving content you explicitly call important (like Subject or certain keywords).


Want me to convert this into functions only (not a class), or add returning subject/from metadata separately?

