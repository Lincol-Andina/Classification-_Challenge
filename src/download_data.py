"""Parse local UCI batch*.dat (format: 'class;conc 1:v ... 128:v') -> data/gas_drift.csv."""
import os, glob
import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uci270", "batch*.dat")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gas_drift.csv")

def parse_file(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            label = int(toks[0].split(";")[0])  # class 1..6
            row = [float(t.split(":")[1]) for t in toks[1:]]
            assert len(row) == 128, (path, len(row))
            X.append(row)
            y.append(label)
    return np.array(X), np.array(y)

def main():
    files = sorted(glob.glob(SRC))
    print(f"Parsing {len(files)} files...")
    Xs, ys = [], []
    for p in files:
        X, y = parse_file(p)
        Xs.append(X); ys.append(y)
        print(f"  {os.path.basename(p)}: {X.shape}")
    X = np.vstack(Xs); y = np.concatenate(ys)
    cols = [f"f{i+1}" for i in range(128)]
    df = pd.DataFrame(X, columns=cols)
    df["class"] = y
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {df.shape} -> {OUT}")
    print(df["class"].value_counts().sort_index())

if __name__ == "__main__":
    main()
