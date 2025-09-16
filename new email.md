import re
import pandas as pd

# ---------------------------
# Lightweight cleaner (same as before)
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


# ---------------------------
# Apply transformation only for NOTETYPE == 'Email In'
# ---------------------------

def process_email_in(df: pd.DataFrame) -> pd.DataFrame:
    df['clean_text'] = None
    df['tokens'] = None
    
    mask = df['NOTETYPE'].str.lower() == 'email in'
    
    df.loc[mask, ['clean_text','tokens']] = df.loc[mask, 'NOTE'].map(clean_email_light).apply(pd.Series)
    
    return df


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Read your full dataframe
    df_case_notes = pd.read_csv("Data/CASE_NOTES_last_30_days.csv")

    df_case_notes = process_email_in(df_case_notes)

    # Show only first 5 cleaned email rows to avoid huge output
    print(df_case_notes[df_case_notes['NOTETYPE'].str.lower() == 'email in'][['CASE_NUMBER','NOTETYPE','clean_text','tokens']].head())
    
    # Optionally save results
    df_case_notes.to_csv("cleaned_CASE_NOTES.csv", index=False)
    print("✅ Cleaned notes saved to cleaned_CASE_NOTES.csv")