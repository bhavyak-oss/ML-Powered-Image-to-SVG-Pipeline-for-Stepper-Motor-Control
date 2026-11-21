"""
eval_ocr.py
-----------
OCR ka Character Error Rate (CER) aur Word Error Rate (WER) nikalta hai.

Chalane ka tarika (project ke main folder me):
    ./venv/bin/python3 -W ignore eval_ocr.py test2.jpeg

Do baar napta hai:
  RAW       = OCR ka seedha output (spelling correction se pehle)      <- asli OCR accuracy
  CORRECTED = vocab.txt se spelling theek hone ke baad                  <- poori pipeline ka output
DHYAN: vocab.txt me isi document ke words hain, isliye CORRECTED wala number ZYADA ACCHA dikhega
(ye test-data leak hai). OCR accuracy batani ho to RAW wala number do.

CER = (substitution + deletion + insertion, characters pe) / ground-truth ke total characters
WER = wahi, words pe.
Ground truth neeche GT me likha hai (upar se neeche, ek line = ek entry). Apni image ke hisaab se badlo,
ya  --gt file.txt  se do (ek line me ek entry).
"""
import argparse, os, sys

GT_DEFAULT = ["Personal", "Responsibility", "Language", "Punctuality", "Mentoring",
              "Self-Management", "Self-Confidence", "Analysis", "Innovativeness", "Enterprising"]


def edit_distance(a, b):
    """Levenshtein distance; a, b = strings ya lists."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer_wer(refs, hyps):
    ref_text, hyp_text = " ".join(refs), " ".join(hyps)
    c = edit_distance(ref_text, hyp_text) / max(1, len(ref_text))
    rw, hw = ref_text.split(), hyp_text.split()
    w = edit_distance(rw, hw) / max(1, len(rw))
    return c, w


def report(name, refs, hyps):
    print(f"\n=== {name} ===")
    for r, h in zip(refs, hyps):
        print(f"  {r:18s} -> {h}")
    for label, f in (("case-sensitive", lambda s: s), ("case-insensitive", str.lower)):
        c, w = cer_wer([f(x) for x in refs], [f(x) for x in hyps])
        print(f"  {label:17s} CER = {c*100:5.1f}%   WER = {w*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--gt", help="ground truth file (ek line me ek entry)")
    a = ap.parse_args()
    gt = [l.strip() for l in open(a.gt, encoding="utf-8") if l.strip()] if a.gt else GT_DEFAULT

    base = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(base, "ocr_module"))
    sys.path.insert(1, os.path.join(base, "ui_export_module"))
    img = os.path.abspath(a.image)
    os.chdir(os.path.join(base, "ocr_module"))
    from pipeline import run_pipeline
    doc = run_pipeline(img, use_finetuned=False)
    els = sorted(doc["text_elements"], key=lambda e: e["y_mm"])
    raw = [e.get("raw_text", e["text"]) for e in els]
    cor = [e["text"] for e in els]
    print(f"ground truth lines: {len(gt)}   OCR lines: {len(els)}")
    if len(els) != len(gt):
        print("  (line count alag hai: line miss/merge hui, ye bhi error me ginti hogi)")
    # lines ki ginti alag ho to bhi sahi chale: gt ko lamba karke zip ki jagah pura text compare hota hai
    n = max(len(gt), len(els))
    gt_p, raw_p, cor_p = (gt + [""] * n)[:n], (raw + [""] * n)[:n], (cor + [""] * n)[:n]
    report("RAW OCR (spelling correction se pehle)", gt_p, raw_p)
    report("CORRECTED (vocab.txt ke baad; leak hai, optimistic)", gt_p, cor_p)


if __name__ == "__main__":
    main()
