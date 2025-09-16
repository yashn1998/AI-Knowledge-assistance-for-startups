Perfect ✅ — so you only want to apply the cleaning on rows where NOTETYPE == 'email in', and leave other rows unchanged.
Here’s the refined version of the code:

import pandas as pd

# --- your cleaning function (from before) ---
import re

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
    txt = txt.replace('\r\n', '\n').replace('\r', '\n')
    txt = re.split(r"(-{2,}\s*Original Message\s*-{2,})|(-{2,}\s*Forwarded message\s*-{2,})", txt, flags=re.I)[0]
    txt = re.split(r"\nOn .* wrote:\n", txt, flags=re.I)[0]

    lines = []
    for line in txt.splitlines():
        l = line.strip()
        skip = False
        for pat in HEADER_KEYWORDS:
            if re.match(pat, l, flags=re.I):
                skip = True
                break
        if skip or re.match(r'^[\-\_]{2,}$', l) or re.match(r'^[\W_]+$', l):
            continue
        if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', l) or 'http' in l.lower() or 'www.' in l.lower():
            continue
        if re.match(r'^(fw:|re:)\s*', l, flags=re.I) or len(l) <= 1:
            continue
        lines.append(l)
    txt = "\n".join(lines)

    txt = re.sub(r'\+?\d[\d\-\s\(\)]{6,}\d', ' ', txt)
    txt = re.sub(r'https?://\S+|www\.\S+', ' ', txt)
    txt = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', txt)

    low = txt.lower()
    for phrase in SIGNATURE_PHRASES:
        idx = low.find(phrase)
        if idx != -1:
            txt = txt[:idx]
            break

    tokens = re.findall(r"[A-Za-z']{2,}", txt)
    tokens = [t.lower() for t in tokens if t.lower() not in stopwords]

    cleaned = " ".join(tokens)
    return cleaned, tokens


# --- Apply transformation only for NOTETYPE == 'email in' ---
def process_notes(df: pd.DataFrame) -> pd.DataFrame:
    # Make copies so we don’t overwrite original unless desired
    df['clean_text'] = None
    df['tokens'] = None

    # Filter only 'email in' rows and apply cleaning
    mask = df['NOTETYPE'].str.lower() == 'email in'
    df.loc[mask, ['clean_text','tokens']] = df.loc[mask, 'NOTE'].map(clean_email_light).apply(pd.Series)

    return df


# --- Example usage ---
if __name__ == "__main__":
    data = {
        "CASE_NUMBER":[1,2,3],
        "NOTETYPE":["email in","phone call","email in"],
        "NOTE":[
            "From: abc@example.com\nSubject: test\nHi team, please check logs.\nThanks",
            "Spoke to customer on call, issue resolved.",
            "Sent: Monday, test@example.com\nSubject: Update\nHello,\nCheck attachment.\nRegards"
        ]
    }
    df = pd.DataFrame(data)

    df = process_notes(df)

    # Show only a few rows to avoid Jupyter crash
    print(df[['CASE_NUMBER','NOTETYPE','NOTE','clean_text','tokens']].head())


---

🔑 Key points

Adds clean_text and tokens columns only for rows where NOTETYPE == 'email in'.

Other rows (like "phone call") stay None.

You can later dropna(subset=['clean_text']) if you only want cleaned emails.



---

Do you also want me to replace the original NOTE column text (cleaned version) instead of adding clean_text & tokens separately?

