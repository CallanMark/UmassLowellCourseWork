
import argparse, csv, re, sys
from pathlib import Path
import nltk, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


SEED = 123


nltk.download("stopwords", quiet=True)
STOP = nltk.corpus.stopwords.words("english")


_URL    = re.compile(r"https?://\S+")
_HANDLE = re.compile(r"@\w+")
_TOKS   = re.compile(r"[#@\w']+")

CLUST = {}  # maps word to CLUST_###
def load_cluster_files(cluster_files):
    """Load word clusters from multiple files.
    
    Args:
        cluster_files: List of paths to cluster files
        
    Returns:
        dict: Mapping of words to cluster IDs
    """
    clusters = {}
    for file_path in cluster_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        parts = line.strip().split()
                        if len(parts) < 2:
                            print(f"[WARN] Skipping invalid line {line_num} in {file_path}", 
                                  file=sys.stderr)
                            continue
                        cid, word = parts[0], parts[1]
                        clusters[word] = f"CLUST_{cid}"
                    except Exception as e:
                        print(f"[ERROR] Failed to process line {line_num} in {file_path}: {e}",
                              file=sys.stderr)
        except FileNotFoundError:
            print(f"[WARN] Cluster file {file_path} not found - skipping",
                  file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}",
                  file=sys.stderr)
    
    if not clusters:
        print("[WARN] No valid cluster files found - proceeding without clusters",
              file=sys.stderr)
    
    return clusters

# Load clusters from default file and any additional files
CLUST = load_cluster_files(["word_clusters.txt"])



def normalize(sent):
    return " ".join(CLUST.get(tok, tok) for tok in _TOKS.findall(clean(sent)))


def clean(text: str) -> str:
    # Lower‑case & replace URLs / user handles with placeholders
    text = _URL.sub(" URL ", text)
    text = _HANDLE.sub(" USER ", text)
    return text.lower()

def load_file(path: Path, test: bool = False):
    ids, labels, texts = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if test:
                tid, txt = parts[0], parts[-1]
                ids.append(tid); texts.append(txt)
            else:
                tid, lab, txt = parts[0], parts[1], parts[-1]
                ids.append(tid); labels.append(lab); texts.append(txt)
    return ids, labels, texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--train", required=True)
    ap.add_argument("-v", "--val",   required=True)
    ap.add_argument("-t", "--test",  required=True)
    args = ap.parse_args()

    # Load data
    tr_ids, y_tr, X_tr_raw = load_file(Path(args.train))
    va_ids, y_va, X_va_raw = load_file(Path(args.val))
    te_ids, _ ,   X_te_raw = load_file(Path(args.test), test=True)

    # Pre‑clean
    X_tr = [clean(t) for t in X_tr_raw]
    X_va = [clean(t) for t in X_va_raw]
    X_te = [clean(t) for t in X_te_raw]

    # Feature definitions
    word_vect = TfidfVectorizer(
        tokenizer=lambda s: normalize(s).split(),
        stop_words=STOP,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True)

    char_vect = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True)

    # Pipeline
    model = Pipeline([
        ("feats", FeatureUnion([("word", word_vect), ("char", char_vect)])),
        ("clf",  LogisticRegression(
                    max_iter=1500, solver='saga',
                    class_weight="balanced",
                    C=2.0, n_jobs=-1, random_state=SEED))
    ])

    # Train
    model.fit(X_tr, y_tr)

    # Validation score
    va_pred = model.predict(X_va)
    bal = balanced_accuracy_score(y_va, va_pred)
    print(f"[INFO] balanced‑accuracy on validation = {bal:.4f}", file=sys.stderr)

    # Predict test
    te_pred = model.predict(X_te)
    with open("predictions.txt", "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        for tid, tag in zip(te_ids, te_pred):
            w.writerow([tid, tag])
    print("[INFO] predictions.txt written.")

if __name__ == "__main__":
    main()
