import pandas as pd

df = pd.DataFrame({
    "id": [1,2,3],
    "text": [
        "John went to New York and met Mary.",
        "This is a sample sentence containing the word foo-bar and baz.",
        None
    ],
    "notes": ["remove foo", "keep it", "Foo is here"]
})

stopwords = ["john", "mary", "new york", "foo", "foo-bar"]

# Example 1: preserve originals, create new columns
out = remove_custom_stopwords_from_df(df, ["text", "notes"], stopwords)
print(out[["text", "text_clean", "notes", "notes_clean"]])

# Example 2: overwrite original columns
out2 = remove_custom_stopwords_from_df(df, ["text"], ["new york", "mary"], preserve_original=False)
print(out2["text"])