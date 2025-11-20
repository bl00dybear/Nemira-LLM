import os, re, json, random, hashlib, argparse, unicodedata
from pathlib import Path
from datasets import load_dataset

try:
    import langid
    HAVE_LANGID = True
except ImportError:
    HAVE_LANGID = False

RATIOS = {
    "mc4": 0.40,
    "wikipedia": 1,
    "cc100": 0.20,
    "europarl": 0.10,
    "jrc_acquis": 0.05,
}

MIN_LEN = 40
MAX_LEN = 3000
ROM_LANG_THRESHOLD = 0.90

def normalize(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u00a0", " ").strip()
    return re.sub(r"\s+", " ", t)

def is_ro(t: str) -> bool:
    if HAVE_LANGID:
        lang, p = langid.classify(t)
        return lang == "ro" and p >= ROM_LANG_THRESHOLD
    markers = ["ă", "â", "î", "ș", "ț", "este", "și", "care"]
    return any(m in t.lower() for m in markers)

def passed_filters(t: str) -> bool:
    if len(t) < MIN_LEN or len(t) > MAX_LEN: return False
    if not any(x in t for x in ".?!"): return False
    if not is_ro(t): return False
    return True

def sha1(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()

def split_paragraphs(raw: str):
    for part in re.split(r"[\r\n]+", raw):
        p = part.strip()
        if p:
            yield p

def load_source(name: str):
    try:
        if name == "mc4":
            return load_dataset("mc4", "ro", split="train", streaming=True), "text"
        if name == "wikipedia":
            return load_dataset("wikimedia/wikipedia", "20231101.ro", split="train"), "text"
        if name == "cc100":
            return load_dataset("cc100", "ro", split="train", streaming=True), "text"
        if name == "europarl":
            return load_dataset("europarl", "ro", split="train"), "text"
        if name == "jrc_acquis":
            return load_dataset("jrc_acquis", "ro", split="train"), "text"
    except Exception as e:
        print(f"[WARN] Sursa {name} indisponibilă: {e}")
    return None, None

USE_LANGID = False 
MAX_PARAS_PER_ARTICLE = 8

def is_ro(t: str) -> bool:
    if USE_LANGID and HAVE_LANGID:
        lang, p = langid.classify(t)
        return lang == "ro" and p >= ROM_LANG_THRESHOLD
    markers = ["ă", "â", "î", "ș", "ț", " este ", " și ", " care "]
    low = t.lower()
    return any(m in low for m in markers)

def sample(name: str, target: int):
    if target <= 0: return []
    print(f"[INFO] Sursa {name} -> țintă {target}")
    ds, field = load_source(name)
    if ds is None:
        print(f"[INFO] Skip {name}")
        return []
    out, seen = [], set()
    it = iter(ds)
    processed_articles = 0
    while len(out) < target:
        try:
            ex = next(it)
        except StopIteration:
            break
        raw = ex.get(field, "")
        if not raw:
            continue
        paras_added = 0
        for para in split_paragraphs(raw):
            if paras_added >= MAX_PARAS_PER_ARTICLE:
                break
            para = normalize(para)
            if not para or not passed_filters(para):
                continue
            h = sha1(para)
            if h in seen:
                continue
            seen.add(h)
            out.append(para)
            paras_added += 1
            if len(out) >= target:
                break
        processed_articles += 1
        if processed_articles % 500 == 0:
            print(f"[INFO] {name}: {len(out)}/{target} (articole {processed_articles})")
    return out

def redistribute(ratios: dict):
    active = {k: v for k, v in ratios.items() if v > 0}
    s = sum(active.values())
    return {k: v / s for k, v in active.items()}

def build(total: int, ratios: dict):
    ratios = redistribute(ratios)
    per = {k: int(v * total) for k, v in ratios.items()}
    all_text = []
    for k, cnt in per.items():
        all_text.extend(sample(k, cnt))
    seen, final = set(), []
    for t in all_text:
        h = sha1(t)
        if h in seen: continue
        seen.add(h)
        final.append(t)
    print(f"[INFO] Total final după dedup: {len(final)}")
    return final

def save(lines, path: str):
    with Path(path).open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
    print(f"[DONE] Salvat {len(lines)} -> {path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/text-data.jsonl")
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    corpus = build(args.total, RATIOS)
    random.shuffle(corpus)
    save(corpus, args.out)

if __name__ == "__main__":
    main()