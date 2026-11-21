"""
eval_ocr_batch.py
-----------------
Kai images pe OCR ka CER / WER nikalta hai aur average deta hai.

Setup (project main folder me):
    eval_data/                 <- naya folder
        img01.jpg   img01.txt  <- image + uski ground truth (jo asli me likha hai, ek line = ek likhi hui line)
        img02.png   img02.txt
        ...
Image aur .txt ka naam same hona chahiye (extension alag). Space wale naam mat rakho.

Chalane ka tarika:
    ./venv/bin/python3 -W ignore eval_ocr_batch.py            # eval_data/ folder
    ./venv/bin/python3 -W ignore eval_ocr_batch.py my_folder  # alag folder

Output: har image ka CER/WER + average, aur eval_results.csv.
RAW      = spelling correction se pehle (asli OCR accuracy).
CORRECTED= vocab.txt ke baad. Naye images me vocab.txt ke words nahi hote, isliye ye number ab leak nahi hai,
           par vocab galat words ko "sudhar" bhi sakta hai, to dono dekho.
Average 2 tarah ke:
  MACRO = har image ke CER ka seedha average (har image barabar).
  POOLED= sab images ke total galtiyan / total ground-truth characters (zyada text wali image ka zyada vajan).
Poster ke liye MACRO ya POOLED me se ek chuno aur likho ki kaunsa hai.
"""
import argparse, csv, os, sys

IMG_EXT = (".jpg", ".jpeg", ".png")


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def errs(refs, hyps, lower=False):
    """(char_edits, char_len, word_edits, word_len) -- lines ko space se jodke."""
    r, h = " ".join(refs), " ".join(hyps)
    if lower:
        r, h = r.lower(), h.lower()
    rw, hw = r.split(), h.split()
    return edit_distance(r, h), len(r), edit_distance(rw, hw), len(rw)


def pct(a, b):
    return 100.0 * a / max(1, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="eval_data")
    a = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.abspath(a.folder)
    sys.path.insert(0, os.path.join(base, "ocr_module"))
    sys.path.insert(1, os.path.join(base, "ui_export_module"))

    items = []
    for f in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in IMG_EXT:
            continue
        gt_path = os.path.join(folder, stem + ".txt")
        if not os.path.exists(gt_path):
            print(f"skip (ground truth {stem}.txt nahi mili): {f}")
            continue
        gt = [l.strip() for l in open(gt_path, encoding="utf-8") if l.strip()]
        items.append((stem, os.path.join(folder, f), gt))
    if not items:
        print("koi (image + .txt) jodi nahi mili:", folder); return
    print(f"{len(items)} images mili. OCR chal raha hai (thoda time lagega)...")

    os.chdir(os.path.join(base, "ocr_module"))
    from pipeline import run_pipeline

    rows, tot = [], {k: [0, 0, 0, 0] for k in ("raw", "raw_ci", "cor", "cor_ci")}
    for stem, path, gt in items:
        try:
            doc = run_pipeline(path, use_finetuned=False)
        except Exception as e:
            print(f"  FAIL {stem}: {type(e).__name__} {e}")
            continue
        els = sorted(doc["text_elements"], key=lambda e: e["y_mm"])
        raw = [e.get("raw_text", e["text"]) for e in els]
        cor = [e["text"] for e in els]
        row = {"image": stem, "gt_lines": len(gt), "ocr_lines": len(els)}
        for key, hyp, low in (("raw", raw, False), ("raw_ci", raw, True), ("cor", cor, False), ("cor_ci", cor, True)):
            ce, cl, we, wl = errs(gt, hyp, low)
            row[key + "_CER"], row[key + "_WER"] = pct(ce, cl), pct(we, wl)
            for i, v in enumerate((ce, cl, we, wl)):
                tot[key][i] += v
        rows.append(row)
        print(f"  {stem:20s} lines {len(gt)}/{len(els)}  RAW CER {row['raw_CER']:5.1f}%  WER {row['raw_WER']:6.1f}%"
              f"   | CORRECTED CER {row['cor_CER']:5.1f}%  WER {row['cor_WER']:6.1f}%")

    n = len(rows)
    if not n:
        return
    print(f"\n===== {n} images ka average =====")
    print(f"{'':28s}{'CER':>8s}{'WER':>9s}")
    for name, key in (("RAW", "raw"), ("RAW (case ignore)", "raw_ci"),
                      ("CORRECTED", "cor"), ("CORRECTED (case ignore)", "cor_ci")):
        macro_c = sum(r[key + "_CER"] for r in rows) / n
        macro_w = sum(r[key + "_WER"] for r in rows) / n
        pool_c, pool_w = pct(tot[key][0], tot[key][1]), pct(tot[key][2], tot[key][3])
        print(f"{name:24s} MACRO {macro_c:6.1f}% {macro_w:7.1f}%")
        print(f"{'':24s} POOLED{pool_c:6.1f}% {pool_w:7.1f}%")
    out = os.path.join(base, "eval_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
