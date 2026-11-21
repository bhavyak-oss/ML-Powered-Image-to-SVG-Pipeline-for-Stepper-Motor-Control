import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "ocr_module")
from PIL import Image, ImageDraw, ImageOps
from ocr_multilang import _engine
p = sys.argv[1]; eng = _engine("hi")
g = ImageOps.autocontrast(Image.open(p).convert("L"), cutoff=2)
g = g.resize((g.width*2, g.height*2)); g.save("/tmp/hindi_pre.png")
big = {"box_thresh": 0.3, "text_score": 0.3, "unclip_ratio": 2.2}
runs = [("default", p, {}, 1), ("bigger", p, big, 1),
        ("pre", "/tmp/hindi_pre.png", {}, .5), ("pre_bigger", "/tmp/hindi_pre.png", big, .5)]
tiles = []
for name, path, kw, s in runs:
    res, _ = eng(path, **kw); res = res or []
    print(f"\n{name}: {len(res)} boxes")
    im = Image.open(p).convert("RGB"); d = ImageDraw.Draw(im)
    for box, text, conf in res:
        pts = [(x*s, y*s) for x, y in box]
        d.polygon(pts, outline="red", width=3)
        print(f"  {float(conf):.2f} y={round(min(q[1] for q in pts))}-{round(max(q[1] for q in pts))} {text}")
    tiles.append(im)
W = max(t.width for t in tiles); H = sum(t.height for t in tiles) + 10*len(tiles)
sheet = Image.new("RGB", (W, H), "white"); y = 0
for t in tiles:
    sheet.paste(t, (0, y)); y += t.height + 10
sheet.save("hindi_boxes_all.png"); print("saved hindi_boxes_all.png")
