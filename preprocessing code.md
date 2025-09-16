Clean and extract useful tokens from email-like text in casenotes['NOTE']

Below is a ready-to-run Python module you can drop into a Jupyter notebook or script. It:

strips typical email headers (From:, Sent:, Subject:, Cc:, -----Original Message-----, On ... wrote: etc.)

removes email addresses, phone numbers, URLs, and short noise tokens

tokenizes, lowercases, removes stopwords and optionally lemmatizes

returns both a cleaned plain text and a token list for each row


There are two flavors in the code: a lightweight version (no heavy deps) and an advanced version (uses nltk for better stopword/lemmatization). Use whichever suits you.

import re
import pandas as pd

# ---------------------------
# Lightweight cleaner (no external deps)
# ---------------------------
BASIC_STOPWORDS = {
    "the","and","is","in","to","of","for","on","with","this","that","a","an","by","from",
    "as","it","be","are","we","you","your","our","at","or","not","have","has","will",
}

HEADER_KEYWORDS = [
    r"^from:", r"^sent:", r"^to:", r"^subject:", r"^cc:", r"^bcc:", r"^date:",
    r"^reply-to:", r"^attachments?:", r"^importance:", r"^priority:", r"^x-",
]

SIGNATURE_PHRASES = [
    "sent from my", "best regards", "regards", "thanks", "thank you", "cheers",
    "kind regards", "warm regards"
]

def clean_email_light(text: str, stopwords=BASIC_STOPWORDS):
    if not isinstance(text, str):
        return "", []

    txt = text

    # Normalize whitespace
    txt = txt.replace('\r\n', '\n').replace('\r', '\n')

    # Remove quoted original message blocks entirely (common separators)
    txt = re.split(r"(-{2,}\s*Original Message\s*-{2,})|(-{2,}\s*Forwarded message\s*-{2,})", txt, flags=re.I)[0]

    # Remove common "On <date> <name> wrote:" quoting (keep the text before such quote)
    txt = re.split(r"\nOn .* wrote:\n", txt, flags=re.I)[0]

    # Remove lines that look like headers
    lines = []
    for line in txt.splitlines():
        l = line.strip()
        skip = False
        # header patterns
        for pat in HEADER_KEYWORDS:
            if re.match(pat, l, flags=re.I):
                skip = True
                break
        # skip lines that are mostly email addresses, long sequences of dashes, or only special chars
        if skip or re.match(r'^[\-\_]{2,}$', l) or re.match(r'^[\W_]+$', l):
            continue
        # skip lines that primarily contain email addresses or "mailto:" or URLs
        if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', l) or 'http' in l.lower() or 'www.' in l.lower():
            continue
        # drop short noise tokens like "FW:", "RE:", or single-character lines
        if re.match(r'^(fw:|re:)\s*', l, flags=re.I) or len(l) <= 1:
            continue
        lines.append(l)
    txt = "\n".join(lines)

    # remove phone numbers (various formats)
    txt = re.sub(r'\+?\d[\d\-\s\(\)]{6,}\d', ' ', txt)

    # remove URLs leftover
    txt = re.sub(r'https?://\S+|www\.\S+', ' ', txt)

    # remove email addresses leftover
    txt = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', txt)

    # remove signature phrases and anything after them (try to cut signature block)
    low = txt.lower()
    for phrase in SIGNATURE_PHRASES:
        idx = low.find(phrase)
        if idx != -1:
            txt = txt[:idx]
            break

    # keep only alphabetic tokens (allow apostrophes), remove digits/punctuation
    tokens = re.findall(r"[A-Za-z']{2,}", txt)
    tokens = [t.lower() for t in tokens if t.lower() not in stopwords]

    cleaned = " ".join(tokens)
    return cleaned, tokens

# ---------------------------
# Advanced cleaner using NLTK (recommended if you want lemmatization / better stopwords)
# ---------------------------

def prepare_nltk():
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except Exception:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)

def clean_email_nltk(text: str):
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
    except Exception as e:
        raise RuntimeError("NLTK not available. Run `pip install nltk` and call prepare_nltk()") from e

    # ensure corpora
    prepare_nltk()

    stopset = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned, tokens = clean_email_light(text, stopwords=set())  # get basic tokens (no stopwords filtered)
    # lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopset]
    cleaned = " ".join(tokens)
    return cleaned, tokens

# ---------------------------
# Example usage on your dataframe `casenotes`
# ---------------------------

if __name__ == "__main__":
    # Example: load or assume you already have casenotes dataframe
    # casenotes = pd.read_csv("casenotes.csv")  # or however you load it

    # Create sample df for demonstration:
    sample = {
        "CASE_NUMBER":[1,2],
        "NOTE":[
            "From: John Doe <john@example.com>\nSent: Tuesday, August 12, 2025 7:12 AM\nSubject: Re: issue\nHi team,\nPlease check the logs. On Tue, Aug 11, Jane wrote:\n> some quoted\nThanks,\nJohn\nSent from my iPhone",
            "Mareille Jacobs <mjacobs@microsoft.com>  To:  team@company.com  Subject: Update\nCall me at +1 (800) 123-4567. https://example.com\n--\nRegards,\nMareille"
        ]
    }
    casenotes = pd.DataFrame(sample)

    # Apply lightweight cleaner
    casenotes['clean_text'], casenotes['tokens'] = zip(*casenotes['NOTE'].map(clean_email_light))

    # If you want to use NLTK advanced cleaner:
    # prepare_nltk()
    # casenotes['clean_text_nltk'], casenotes['tokens_nltk'] = zip(*casenotes['NOTE'].map(clean_email_nltk))

    print(casenotes[['NOTE','clean_text','tokens']].to_string(index=False))

Notes & tips

The clean_email_light function is robust and fast — good for pipelines where you do not want heavy downloads.

If you need better linguistic normalization (lemmatization) and more accurate stopword removal, use the clean_email_nltk path (run pip install nltk and prepare_nltk() once).

If your "relevant tokens" are not just words (e.g., you want to keep case numbers, IPs, product codes), adjust the token regex and the stopword set accordingly.

If you want to keep sentence structure (not just tokens) you can return txt after header removal rather than token-joining.

For email thread removal it's usually good to: split on "-----Original Message-----", "On <date> ... wrote:", or common reply separators; the code already does that.


If you want, I can:

adapt the regex to your exact sample (paste one real NOTE value), or

produce an implementation that extracts structured fields (phone numbers, case IDs, dates) into separate columns.


