# scripts/inspect_probs.py
import pandas as pd

# paths (adjust if needed)
probs_csv = r"checkpoint\Aburaage_probs.csv"
ingredient_map = r"data\metadata\ingredient_list_label.txt"  # format "idx,ingredient name" per line

# load probs
df = pd.read_csv(probs_csv)

# load ingredient names into list where index matches prob_i
ing = []
with open(ingredient_map, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # handle "idx,name" or "name" formats
        parts = line.split(",", 1)
        if len(parts) == 2:
            ing.append(parts[1].strip())
        else:
            ing.append(parts[0].strip())

# helper to get top-k names for a row
def topk_for_row(row_idx, k=10, threshold=None):
    row = df.iloc[row_idx].values  # numpy array of probabilities
    # get indices sorted by probability desc
    order = row.argsort()[::-1]
    result = []
    for idx in order[:k]:
        p = float(row[idx])
        if threshold is not None and p < threshold:
            continue
        name = ing[idx] if idx < len(ing) else f"idx_{idx}"
        result.append((idx, name, p))
    return result

if __name__ == "__main__":
    nrows = len(df)
    print(f"Loaded {nrows} rows, {df.shape[1]} ingredient columns.")
    # show top-5 for first few rows
    for i in range(min(5, nrows)):
        print(f"\nRow {i} top-10 predictions:")
        for idx, name, p in topk_for_row(i, k=10):
            print(f"  {idx:3d}  {p:0.6f}  {name}")