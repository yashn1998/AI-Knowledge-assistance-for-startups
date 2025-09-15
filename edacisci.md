Nice — below is a step-by-step Jupyter Notebook you can copy-paste cell-by-cell and run.
This notebook is focused on discovering what each dataframe contains and precisely how the 4 datasets link to each other (exact keys, coverage %, candidate keys, fuzzy linking using serials/product+date, and quick visual of links).

It assumes your four DataFrames are already loaded in the notebook as:

df_case_notes

df_case_headers

df_issue_insights

df_rma


If you load from CSVs instead, edit the CSV_PATHS in the first code cell.


---

0 — (Markdown)

# EDA: Identify dataset roles & how the 4 Cisco tables link
Goal: For each table find role (what it stores), candidate primary key(s), important columns,
and compute how tables connect (exact FK coverage + fuzzy linking).
Run cells sequentially.


---

1 — Install & imports (Code)

# Run if you need libs. You can skip if already installed.
!pip install -q pandas numpy matplotlib seaborn scikit-learn rapidfuzz networkx

# Imports
import os, re, json, itertools
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import process, fuzz
import networkx as nx

sns.set(style="whitegrid", rc={"figure.figsize": (10,4)})
pd.set_option('display.max_colwidth', 250)


---

2 — Load data (Code)

# Edit CSV paths if needed. If you already have dfs in memory, they'll be picked up.
CSV_PATHS = {
    "case_notes": "case_notes.csv",
    "case_headers": "case_headers.csv",
    "issue_insights": "issue_insights.csv",
    "rma": "rma.csv"
}

dfs = {}
# Try to get in-memory variables first
for name in ["case_notes","case_headers","issue_insights","rma"]:
    if name in globals():
        dfs[name] = globals()[name]

# Load CSVs for missing ones
for k,p in CSV_PATHS.items():
    if k not in dfs and os.path.exists(p):
        print("Loading", p)
        dfs[k] = pd.read_csv(p, low_memory=False)

print("Loaded:", list(dfs.keys()))


---

3 — Generic table summarizer (Code)

def summarize_table(df, name, top_n=10):
    print("\n" + "="*60)
    print(f"TABLE: {name}  |  shape: {df.shape}")
    print("="*60)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nNull counts (top):")
    print(df.isnull().sum().sort_values(ascending=False).head(20))
    print("\nSample columns & top values (up to 20):")
    for col in df.columns[:50]:  # only first 50 cols for display safety
        nv = df[col].nunique(dropna=False)
        print(f" - {col}  |  dtype: {df[col].dtype}  |  unique: {nv}")
        if nv <= top_n or df[col].dtype == object:
            print("    top:", df[col].value_counts(dropna=False).head(5).to_dict())
    # cardinality hints
    approx_pk = [c for c in df.columns if df[c].nunique() >= 0.9*len(df)]
    if approx_pk:
        print("\nColumns that look nearly-unique (candidate PKs):", approx_pk)
    print("\nFirst rows:")
    display(df.head(3))

# Run for each DF
for k, df in dfs.items():
    summarize_table(df, k)


---

4 — Candidate key finder (Code)

def candidate_keys(df, min_uniqueness=0.8):
    n = len(df)
    cand = []
    for c in df.columns:
        try:
            uniq = df[c].nunique(dropna=False)
            ratio = uniq / max(1,n)
            if ratio >= min_uniqueness:
                cand.append((c, uniq, ratio))
        except Exception:
            pass
    return sorted(cand, key=lambda x: -x[2])

for name, df in dfs.items():
    print("\n", name, "candidate keys (>=80% unique):")
    print(candidate_keys(df, 0.8))


---

5 — Shared column discovery across tables (Code)

# Which column names are shared across tables?
cols_per_table = {k:set(df.columns) for k,df in dfs.items()}
all_cols = set().union(*cols_per_table.values())
shared = {col: [t for t in cols_per_table if col in cols_per_table[t]] for col in all_cols}
shared_cols = {c:tbls for c,tbls in shared.items() if len(tbls)>1}
print("Columns present in more than one table (name -> tables):")
for c,tbls in sorted(shared_cols.items()):
    print(f"{c} -> {tbls}")


---

6 — Exact join / FK coverage analysis (Code)

# Helper to compute coverage when joining child -> parent by columns
def fk_coverage(parent_df, parent_col, child_df, child_col):
    parent_vals = set(parent_df[parent_col].dropna().astype(str).unique())
    child_vals = child_df[child_col].dropna().astype(str)
    total_child = len(child_vals)
    if total_child == 0:
        return {"total_child":0, "matches":0, "pct_match":None}
    matches = child_vals.isin(parent_vals).sum()
    return {"total_child":int(total_child), "matches":int(matches), "pct_match":float(matches/total_child)}

# Try exact FK coverage on plausible pairs
key_candidates = ["CASE_NUMBER","CASE_NUM","CASEID","CASE_ID","RMA_NUMBER","SERNUM","SER_NUM","SERIAL_NUMBER","ITEM_NAME_RMA","PRODUCT_FAMILY","FAILURE_CODE"]
pairs = []
for parent_name, parent in dfs.items():
    for child_name, child in dfs.items():
        if parent_name == child_name: continue
        for pc in parent.columns:
            for cc in child.columns:
                if pc.upper() == cc.upper():  # same column name
                    cov = fk_coverage(parent, pc, child, cc)
                    pairs.append((parent_name, child_name, pc, cc, cov["matches"], cov["total_child"], cov["pct_match"]))
# Show pairs with meaningful coverage
pairs_sorted = sorted([p for p in pairs if p[6] is not None], key=lambda x: -x[6])  # sort by pct_match desc
for p in pairs_sorted[:50]:
    print(f"Parent:{p[0]} Col:{p[2]} <- Child:{p[1]} Col:{p[3]}  | matches {p[4]}/{p[5]}  pct={p[6]:.3f}")


---

7 — For each pair of tables: best exact join columns & coverage (Code)

# For each ordered pair (parent, child), find top 5 matching column-name joins by coverage
def best_exact_joins(parent_df, child_df, topk=5):
    results = []
    for pc in parent_df.columns:
        for cc in child_df.columns:
            if pc.upper() == cc.upper():
                cov = fk_coverage(parent_df, pc, child_df, cc)
                if cov["pct_match"] is not None:
                    results.append((pc, cc, cov["matches"], cov["total_child"], cov["pct_match"]))
    return sorted(results, key=lambda x:-x[4])[:topk]

for parent_name, parent_df in dfs.items():
    for child_name, child_df in dfs.items():
        if parent_name == child_name: continue
        best = best_exact_joins(parent_df, child_df)
        if best:
            print(f"\n{parent_name} <- {child_name}  (top exact joins):")
            for b in best:
                print(" ", b)


---

8 — Cardinality check for primary joins (Code)

# For a given join column pair, compute cardinality pattern: 1-to-1, 1-to-many, many-to-1, many-to-many
def cardinality_pattern(parent_df, parent_col, child_df, child_col):
    # drop nulls and cast to str
    p = parent_df[[parent_col]].dropna().astype(str)
    c = child_df[[child_col]].dropna().astype(str)
    # how many child records per parent key (using only keys present in parent)
    parent_keys = set(p[parent_col].unique())
    c_in_parent = c[c[child_col].isin(parent_keys)]
    if c_in_parent.empty:
        return None
    counts = c_in_parent.groupby(child_col).size().describe().to_dict()
    return counts

# Example: run for CASE_NUMBER where present
for pname, pdf in dfs.items():
    for cname, cdf in dfs.items():
        if pname==cname: continue
        if "CASE_NUMBER" in pdf.columns and "CASE_NUMBER" in cdf.columns:
            pattern = cardinality_pattern(pdf, "CASE_NUMBER", cdf, "CASE_NUMBER")
            print(pname, "<-", cname, "cardinality summary:", pattern)


---

9 — Fuzzy linking attempts when exact join fails

We try linking using likely alternative keys:

Serial numbers (SERNUM,SER_NUM,SERIAL_NUMBER) — fuzzy exact

Combination product_family + date proximity: candidate join for RMAs → headers/notes (use date window)


9.1 Fuzzy serial match (Code)

# Gather candidate serial column names
serial_cols = []
for name, df in dfs.items():
    for c in df.columns:
        if any(k in c.upper() for k in ["SER","SERIAL","SERNUM","SER_NUM","SN"]):
            serial_cols.append((name,c))
serial_cols

# If we found serial columns, attempt fuzzy match sample between two tables
def fuzzy_match_column_pairs(df_left, col_left, df_right, col_right, sample_size=500, score_cutoff=90):
    left_vals = df_left[col_left].dropna().astype(str).unique()
    right_vals = df_right[col_right].dropna().astype(str).unique()
    left_sample = list(left_vals[:sample_size])
    matches = []
    # use rapidfuzz.process.extractOne for speed
    for l in left_sample:
        best = process.extractOne(l, right_vals, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= score_cutoff:
            matches.append((l, best[0], best[1]))
    return matches

# Try matching serials across any two tables found
for (t1,c1), (t2,c2) in itertools.permutations(serial_cols, 2):
    print(f"\nFuzzy serial match: {t1}.{c1} -> {t2}.{c2}")
    m = fuzzy_match_column_pairs(dfs[t1], c1, dfs[t2], c2, sample_size=300, score_cutoff=85)
    print("Matches found (sample 10):", len(m), m[:10])


---

9.2 Product + date window linking (RMA -> case_headers/notes) (Code)

# This approach attempts to match RMA rows to case rows by same product and date proximity.
# Choose which columns look like product (PRODUCT_FAMILY, ITEM_NAME_RMA, PRODUCT_NAME) and date columns.

def link_by_product_and_date(df_src, product_cols_src, date_col_src,
                             df_tgt, product_cols_tgt, date_col_tgt,
                             days_window=7, sample_size=500):
    # Create normalized product token for each side by concatenating candidate product columns
    def make_prod(df, cols):
        available = [c for c in cols if c in df.columns]
        if not available:
            return None
        s = df[available[0]].fillna("").astype(str)
        for c in available[1:]:
            s += " " + df[c].fillna("").astype(str)
        return s.str.lower().str.replace(r'[^a-z0-9 ]',' ', regex=True).str.strip()
    src_prod = make_prod(df_src, product_cols_src)
    tgt_prod = make_prod(df_tgt, product_cols_tgt)
    if src_prod is None or tgt_prod is None:
        return None
    # parse dates
    df_src = df_src.copy()
    df_tgt = df_tgt.copy()
    if date_col_src in df_src.columns:
        df_src[date_col_src] = pd.to_datetime(df_src[date_col_src], errors='coerce')
    if date_col_tgt in df_tgt.columns:
        df_tgt[date_col_tgt] = pd.to_datetime(df_tgt[date_col_tgt], errors='coerce')

    # we'll try for a sample
    src_idx = df_src.dropna(subset=[date_col_src]).index[:sample_size]
    matches = 0
    attempts = 0
    for i in src_idx:
        attempts += 1
        pd_src = src_prod.iloc[i]
        dt_src = df_src.loc[i, date_col_src]
        # candidate targets with same product token (contains)
        cond = tgt_prod.str.contains(pd_src.split()[0]) if pd_src else pd.Series(False, index=df_tgt.index)
        if cond.sum()==0:
            continue
        # narrow by date window
        date_low = dt_src - pd.Timedelta(days=days_window)
        date_high = dt_src + pd.Timedelta(days=days_window)
        tgt_window = df_tgt[cond & (df_tgt[date_col_tgt].between(date_low, date_high, inclusive='both'))]
        if len(tgt_window)>0:
            matches += 1
    return {"attempts": attempts, "matches": matches, "pct": matches/max(1,attempts)}

# Example usage — try to link RMA -> case_headers
if "rma" in dfs and "case_headers" in dfs:
    print("RMA -> case_headers product+date linking attempt:")
    res = link_by_product_and_date(dfs["rma"], ["ITEM_NAME_RMA","FAILURE_CODE"], "RMA_DATE",
                                   dfs["case_headers"], ["PRODUCT_NAME","PRODUCT_FAMILY","PROBLEM_CODE"], "EDWSF_CREATE_DTM",
                                   days_window=14, sample_size=300)
    print(res)


---

10 — Create linkage graph & visualize (Code)

# Build a small graph where nodes are tables and edges weighted by best exact join pct (from step 7)
G = nx.DiGraph()
for t in dfs.keys():
    G.add_node(t)

# find best exact join pct for each ordered pair
for parent_name, parent_df in dfs.items():
    for child_name, child_df in dfs.items():
        if parent_name == child_name: continue
        best = best_exact_joins(parent_df, child_df, topk=1)
        if best:
            pc, cc, matches, total, pct = best[0]
            if pct is not None and pct>0:
                G.add_edge(parent_name, child_name, weight=pct, label=f"{pc}<-{cc} {pct:.2f}")

# draw graph
pos = nx.spring_layout(G, seed=2)
plt.figure(figsize=(8,6))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", arrowsize=20)
edge_labels = nx.get_edge_attributes(G,'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title("Table linkage (exact join pct) - edges show parent<-child join and pct matched")
plt.show()


---

11 — Produce a machine-readable linkage report (Code)

report = {"tables":{},"links": []}
for name, df in dfs.items():
    report["tables"][name] = {
        "shape": df.shape,
        "cols": list(df.columns[:200]),
        "approx_candidate_keys": [c for c, _, _ in candidate_keys(df, 0.95)]
    }

# add links found by exact join top candidates
for parent_name, parent_df in dfs.items():
    for child_name, child_df in dfs.items():
        if parent_name==child_name: continue
        best = best_exact_joins(parent_df, child_df, topk=5)
        for pc, cc, matches, total, pct in best:
            if pct is not None and pct>0:
                report["links"].append({
                    "parent": parent_name,
                    "parent_col": pc,
                    "child": child_name,
                    "child_col": cc,
                    "matches": int(matches),
                    "child_total": int(total),
                    "pct": float(pct)
                })

# Save
with open("linkage_report.json","w") as f:
    json.dump(report, f, indent=2, default=str)
print("Saved linkage_report.json — review 'links' array for which columns connect tables and coverage")


---

12 — Practical next actions & interpretation (Markdown)

After running the notebook:
1. Open `linkage_report.json` — it lists likely links (parent.col <- child.col) and the % coverage.
   - If you see CASE_NUMBER joins with pct ~0.8 or higher, that is your primary join path.
   - If CASE_NUMBER coverage is low, check serial columns and product+date fuzzy linking outputs.

2. Check candidate keys printed earlier — these are columns that behave like a PK.
   - If one table has a high-uniqueness column (e.g. RMA_NUMBER / SERNUM), use it as its PK.

3. Inspect fuzzy serial matches and product+date link pct results to measure how many RMAs / issues can be mapped to cases when exact keys are missing.

4. If many-to-many or low coverage observed:
   - Create a canonicalization step: normalize case numbers, trim leading zeros, upper-case product codes, remove punctuation from serials.
   - Use fuzzy matching (rapidfuzz) on normalized keys with a threshold tuned to your data.

5. Use the linkage graph to design your agent context retrieval:
   - For answering a case question: retrieve from case_headers -> case_notes -> issue_insights -> rma using the strongest join path (highest pct).
   - If coverage gaps exist, augment retrieval with fuzzy joins (serial or product+date) as a fallback.

If you want, I can:
- Convert this into a `.ipynb` file for download.  
- Run a small sample analysis (if you paste a tiny sample of each df or their `.head().to_dict()` here).  
- Create canonicalization / normalization code for the specific column names you have (I can auto-generate it after you confirm a few column names like your exact case id and serial columns).


---

If you want the notebook file .ipynb, say “save as notebook” and I’ll create it and give you the download link.

