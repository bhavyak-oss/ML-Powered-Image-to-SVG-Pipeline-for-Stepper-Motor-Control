import warnings; warnings.filterwarnings("ignore")
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageOps

IMG = "T5.png"
eng = RapidOCR()

def run(label, path, **kw):
    try:
        res, _ = eng(path, **kw)
    except Exception as e:
        print(f"\n{label}: ERROR {e}"); return
    res = res or []
    print(f"\n{label}: {len(res)} lines")
    for box, text, conf in res:
        print(f"   {text!r}  conf={float(conf):.2f}")

run("1) default", IMG)
run("2) lower thresholds", IMG, box_thresh=0.3, text_score=0.3)
run("3) bigger boxes", IMG, box_thresh=0.3, text_score=0.3, unclip_ratio=2.2)

g = ImageOps.autocontrast(Image.open(IMG).convert("L"), cutoff=2)
g = g.resize((g.width * 2, g.height * 2))
g.save("/tmp/T5_pre.png")
run("4) autocontrast + 2x upscale", "/tmp/T5_pre.png")
