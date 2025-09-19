Nice — below is a single, reusable function that updates the specified text columns in place on a DataFrame. It:

loads a spaCy model (you can use en_core_web_sm or a transformer like en_core_web_trf),

removes the masking tokens shown in your screenshot (<EMAIL>, <PHONE>, <URL>, <IP>, <CREDIT>, <SSN>, <DATE>, <TIME> and variants),

removes common greetings (configurable),

removes special characters,

removes “unknown” words (tokens mostly made of non-alphanumeric / replacement / control characters),

optionally removes PERSON names (spaCy NER),

lemmatizes tokens and returns cleaned text (lowercased),

writes cleaned text back to the same columns (no new columns created).


Copy/ paste and run in your environment. If you prefer different greeting words, stopword behavior, or a more/less strict “unknown word” rule, tell me and I’ll tune it.

import re
from typing import List, Iterable, Optional
import pandas as pd
import spacy
from spacy.language import Language

# -----------------------
# Helper: install note
# -----------------------
"""
Notes:
- Install spaCy and a model before running:
  pip install spacy
  python -m spacy download en_core_web_sm

  For transformer model:
  pip install spacy[transformers] torch torchvision
  python -m spacy download en_core_web_trf

- Use spacy_model='en_core_web_trf' if you want transformer-based processing (slower, more accurate NER/lemmatization).
"""

# -----------------------
# The function
# -----------------------
def preprocess_and_lemmatize_columns(
    df: pd.DataFrame,
    text_columns: List[str],
    spacy_model: str = "en_core_web_sm",
    remove_person_names: bool = True,
    greetings: Optional[Iterable[str]] = None,
    extra_mask_tokens: Optional[Iterable[str]] = None,
    batch_size: int = 64,
    disable_components: Optional[List[str]] = None,
    inplace: bool = True
) -> pd.DataFrame:
    """
    Clean + lemmatize text columns in-place on the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (modified in-place if inplace=True).
    text_columns : List[str]
        List of column names to process (must exist in df).
    spacy_model : str
        spaCy model name to load (e.g., "en_core_web_sm" or "en_core_web_trf").
    remove_person_names : bool
        If True, removes PERSON entities detected by spaCy.
    greetings : Iterable[str] or None
        Iterable of greeting words/phrases to remove (case-insensitive). If None,
        a sensible default list is used.
    extra_mask_tokens : Iterable[str] or None
        Extra tokens/phrases to treat as mask tokens and remove.
    batch_size : int
        nlp.pipe batch size.
    disable_components : list or None
        spaCy components to disable for speed (e.g., ["tagger","parser"]).
        NER must be enabled if remove_person_names=True.
    inplace : bool
        If True modify df in-place and return it. If False, work on a copy.
    """
    # Validate
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataframe: {missing}")

    if not inplace:
        df = df.copy()

    # Default greetings
    if greetings is None:
        greetings = {
            "hi", "hello", "hey", "dear", "greetings", "good morning", "good afternoon",
            "good evening", "regards", "thanks", "thank you", "thankyou", "yours sincerely",
            "yours faithfully", "sincerely"
        }
    greetings_lower = {g.lower() for g in greetings}

    # Default mask tokens from your screenshot and common variants
    default_masks = {
        "<email>", "email", "<phone>", "phone", "<url>", "url",
        "<ip>", "ip", "<credit>", "credit", "<ssn>", "ssn",
        "<date>", "date", "<time>", "time"
    }
    extra_mask_tokens = set(extra_mask_tokens or ())
    mask_tokens = default_masks.union({t.lower() for t in extra_mask_tokens})

    # Build regex to quickly remove angle-bracketed masks and similar tokens.
    # We'll remove tokens like "<EMAIL>", "EMAIL", "[EMAIL]" (case-insensitive)
    mask_pattern = re.compile(
        r"(?:<|\[)?\b(?:" + "|".join(re.escape(t.strip("<>[]")) for t in mask_tokens) + r")\b(?:>|])?",
        flags=re.IGNORECASE
    )

    # Regex to remove special chars (keep ASCII letters and digits and spaces)
    special_char_re = re.compile(r"[^A-Za-z0-9\s]")

    # Helper to decide if a word is "unknown" (too many non-alnum / control / replacement char)
    def is_unknown_token(tok: str) -> bool:
        if not tok:
            return True
        # replacement character
        if "\ufffd" in tok:
            return True
        # control characters
        if any(ord(c) < 32 for c in tok):
            return True
        alnum_count = sum(1 for c in tok if c.isalnum())
        # if token is mostly non-alphanumeric, consider unknown
        return (alnum_count / max(1, len(tok))) < 0.5

    # Load spaCy model with appropriate disabled components for performance.
    # Ensure NER is kept if removing PERSON names.
    disable = disable_components[:] if disable_components else []
    if remove_person_names and "ner" in disable:
        disable.remove("ner")
    try:
        nlp: Language = spacy.load(spacy_model, disable=disable)
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{spacy_model}' not found. Install it first. See comments above. Error: {e}"
        )

    # We'll use nlp.pipe for speed
    # Build a generator of original texts and process them in order for each column.
    for col in text_columns:
        series = df[col]
        # Prepare texts: convert NaN to empty string but remember mask for restoring if needed
        texts = []
        null_mask = series.isna()
        for v in series:
            if pd.isna(v):
                texts.append("")  # placeholder
            else:
                texts.append(str(v))

        cleaned_texts = []
        # Step 1: remove mask tokens first (quick regex)
        pre_mask_removed = [mask_pattern.sub(" ", t) for t in texts]

        # Now pipe through spaCy for NER + tokenization + lemmatization
        # For memory: disable expensive components already handled above
        for doc in nlp.pipe(pre_mask_removed, batch_size=batch_size, n_process=1):
            if len(doc) == 0:
                cleaned_texts.append("")
                continue

            # If remove_person_names: mark character spans to drop
            remove_spans = []
            if remove_person_names:
                for ent in doc.ents:
                    if ent.label_.upper() == "PERSON":
                        remove_spans.append((ent.start_char, ent.end_char))

            # We'll filter token by token
            kept_tokens = []
            for token in doc:
                # skip if this token lies inside a PERSON entity (if requested)
                if remove_person_names and token.ent_type_ == "PERSON":
                    continue

                # skip punctuation and spaces
                if token.is_space or token.is_punct:
                    continue

                # token text lower and lemma
                lemma = token.lemma_.strip()
                if not lemma:
                    lemma = token.text.strip()
                lemma_lower = lemma.lower()

                # remove greetings anywhere
                # for multi-word greetings (like "good morning"), spaCy tokens "good" and "morning" will be seen separately.
                # We remove any token that exactly matches a greeting token.
                if lemma_lower in greetings_lower:
                    continue

                # remove mask tokens leftover (if any)
                if lemma_lower in mask_tokens:
                    continue

                # Remove special characters inside token
                cleaned_token = special_char_re.sub("", lemma_lower)

                # Drop tokens that become empty after removing special characters
                if not cleaned_token:
                    continue

                # Remove unknown tokens (mostly non-alpha ones, control chars etc)
                if is_unknown_token(cleaned_token):
                    continue

                # Optionally skip pure stopwords? (not requested — so we keep them)
                # Keep token
                kept_tokens.append(cleaned_token)

            # join tokens with single space; lowercased already
            cleaned_text = " ".join(kept_tokens).strip()
            cleaned_texts.append(cleaned_text)

        # Put values back into dataframe, restoring NaN where original was NaN
        out_series = pd.Series(cleaned_texts, index=series.index)
        out_series[null_mask] = pd.NA
        df[col] = out_series

    return df

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # tiny example
    data = {
        "id": [1,2,3],
        "problem_description": [
            "Hello John, I received an email <EMAIL> and my phone <PHONE> is not working! :-)",
            "Hi there — this is a sample. CREDIT card <CREDIT> number shown. \ufffd odd chars.",
            None
        ]
    }
    df_example = pd.DataFrame(data)
    print("Before:")
    print(df_example["problem_description"].tolist())

    preprocess_and_lemmatize_columns(
        df_example,
        ["problem_description"],
        spacy_model="en_core_web_sm",
        remove_person_names=True
    )

    print("\nAfter:")
    print(df_example["problem_description"].tolist())

Notes & tips

By default this uses en_core_web_sm. If you want better NER/lemmatization accuracy use en_core_web_trf (slower, heavier).

disable_components can speed up processing; e.g. ["parser","tagger"] — but do not disable "ner" if remove_person_names=True.

The unknown-token rule removes tokens with less than 50% alphanumeric characters — adjust the threshold inside is_unknown_token for looser/stricter filtering.

If you want to also remove standard stopwords (like "the", "is"), add a check if token.is_stop: continue.

If you have very large datasets, consider processing column-by-column and using n_process in nlp.pipe (requires spaCy v3+ and careful memory settings).


If you want, I can:

extend this to also remove organization/location entities,

output a separate “debug” column with tokens removed (for auditing),

or provide a version that preserves punctuation like '-' inside tokens (e.g., foo-bar). Which would you prefer?


