def _remove_filenames(self, text: str) -> str:
    """
    Remove filenames with common file extensions (including complex ones like logs, tgz, tar.gz).
    """
    # Common extensions + system dump/log/archive patterns
    file_exts = r"(?:pdf|docx?|xls[xm]?|pptx?|zip|tar|tgz|gz|log|txt|cfg|ini|json|xml|jpeg?|png|bmp|svg)"

    # Remove paths like /home/.../file.ext or C:\Users\...\file.ext
    text = re.sub(r"(?:[A-Za-z]:\\|/)[^\s]+?\.(%s)\b" % file_exts, "", text, flags=re.IGNORECASE)

    # Remove dump-like filenames: words/numbers/hyphens ending with extension
    text = re.sub(r"\b[\w\-]+(?:\.[\w\-]+)*\.(%s)\b" % file_exts, "", text, flags=re.IGNORECASE)

    # Extra: remove quoted "Filename: something.ext"
    text = re.sub(r"\bFilename:\s*[\w\-.]+(?:\.[\w\-.]+)+", "", text, flags=re.IGNORECASE)

    return text