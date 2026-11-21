import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "ocr_module")
from ocr_multilang import run_ocr

for r in run_ocr(sys.argv[1], "hi"):
    print(round(r["confidence"], 2), r["text"])
