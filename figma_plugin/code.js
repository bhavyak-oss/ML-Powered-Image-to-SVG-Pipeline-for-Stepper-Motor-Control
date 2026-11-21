// code.js - Figma ke sandbox me chalta hai (figma object milta hai, DOM nahi).
// UI (ui.html) se sirf messages ke through baat hoti hai.

figma.showUI(__html__, { width: 340, height: 600, themeColors: true });

const LOW_CONF = 0.85;                       // isse kam confidence = reviewer ko amber me dikhega
const COLORS = {
  normal: { r: 0.1, g: 0.1, b: 0.1 },
  low: { r: 0.71, g: 0.33, b: 0.04 },        // amber: OCR ko shak hai
  traced: { r: 0.15, g: 0.39, b: 0.92 },     // blue: non-Latin, image se trace hoga
};

let sourceInfo = null;   // jis selection ko scan kiya
let reviewId = null;     // review frame ki id (is session me)

function fail(message) {
  figma.notify(message, { error: true });
  figma.ui.postMessage({ type: 'error', message });
}

async function tryFonts(candidates) {
  for (const f of candidates) {
    try { await figma.loadFontAsync(f); return f; } catch (e) { /* next */ }
  }
  return null;
}

// Review frame dhoondo: pehle selection ke andar/upar, warna last bana hua.
async function findReviewFrame() {
  const sel = figma.currentPage.selection[0];
  let n = sel || null;
  while (n && n.type !== 'PAGE') {
    if (n.getPluginData && n.getPluginData('sessionId')) return n;
    n = n.parent;
  }
  if (reviewId) {
    const node = await figma.getNodeByIdAsync(reviewId);
    if (node && node.type === 'FRAME') return node;
  }
  return null;
}

figma.ui.onmessage = async (msg) => {
  try {
    // 1) UI ne 'scan' bola -> selection ko PNG banakar UI ko do
    if (msg.type === 'scan') {
      const sel = figma.currentPage.selection[0];
      if (!sel) return fail('Select a frame or image first.');
      const bb = sel.absoluteBoundingBox || { x: sel.x, y: sel.y, width: sel.width, height: sel.height };
      const bytes = await sel.exportAsync({ format: 'PNG', constraint: { type: 'SCALE', value: 2 } });
      sourceInfo = { x: bb.x, y: bb.y, width: bb.width, height: bb.height, name: sel.name };
      figma.ui.postMessage({ type: 'png', bytes, name: sel.name });
      return;
    }

    // 2) Server ka OCR result aaya -> editable text layers ka review frame banao
    if (msg.type === 'render') {
      const r = msg.result;
      if (!sourceInfo) return fail('Scan a selection first.');
      const k = sourceInfo.width / r.image_px.width;   // image px -> Figma units

      const frame = figma.createFrame();
      frame.name = 'OCR review - ' + sourceInfo.name;
      frame.resize(sourceInfo.width, sourceInfo.height);
      frame.x = sourceInfo.x + sourceInfo.width + 80;
      frame.y = sourceInfo.y;
      frame.fills = [{ type: 'SOLID', color: { r: 1, g: 1, b: 1 } }];
      frame.clipsContent = true;
      frame.setPluginData('sessionId', r.session_id);

      const latin = await tryFonts([{ family: 'Inter', style: 'Regular' }]);
      const indic = await tryFonts([
        { family: 'Noto Sans Devanagari', style: 'Regular' },
        { family: 'Noto Sans', style: 'Regular' },
      ]);
      if (!latin) return fail('Inter font could not be loaded.');

      let low = 0;
      for (const el of r.elements) {
        const t = figma.createText();
        frame.appendChild(t);
        t.fontName = el.traced && indic ? indic : latin;   // fontName pehle, phir characters
        t.characters = el.text;
        const h = (el.bbox[3] - el.bbox[1]) * k;
        t.fontSize = Math.max(6, Math.round(h * 0.8));
        t.x = el.bbox[0] * k;
        t.y = el.bbox[1] * k;
        t.name = el.id;
        t.setPluginData('elId', el.id);
        const isLow = el.confidence < LOW_CONF;
        if (isLow) low++;
        const color = el.traced ? COLORS.traced : isLow ? COLORS.low : COLORS.normal;
        t.fills = [{ type: 'SOLID', color }];
      }

      reviewId = frame.id;
      figma.currentPage.selection = [frame];
      figma.viewport.scrollAndZoomIntoView([frame]);
      figma.ui.postMessage({ type: 'rendered', count: r.elements.length, low, sessionId: r.session_id });
      return;
    }

    // 3) UI ne 'collect' bola -> review frame ke text layers padho
    if (msg.type === 'collect') {
      const frame = await findReviewFrame();
      if (!frame) return fail('Select the OCR review frame (or scan again).');
      const texts = frame.findAll((n) => n.type === 'TEXT' && n.getPluginData('elId') !== '');
      const edits = {};
      for (const t of texts) edits[t.getPluginData('elId')] = t.characters;
      figma.ui.postMessage({ type: 'texts', edits, sessionId: frame.getPluginData('sessionId') });
      return;
    }

    // 4) Plot-ready SVG ko canvas pe vector bana ke rakho
    if (msg.type === 'svg') {
      const node = figma.createNodeFromSvg(msg.svg);
      node.name = 'Plot-ready SVG';
      if (msg.place === 'source' && sourceInfo) {        // trace mode: image ke bagal me
        node.x = sourceInfo.x + sourceInfo.width + 80;
        node.y = sourceInfo.y;
      } else {
        const frame = await findReviewFrame();
        if (frame) { node.x = frame.x + frame.width + 80; node.y = frame.y; }
      }
      figma.currentPage.selection = [node];
      figma.viewport.scrollAndZoomIntoView([node]);
      figma.notify('Plot-ready SVG added to the canvas');
      return;
    }
  } catch (e) {
    fail('Error: ' + (e && e.message ? e.message : String(e)));
  }
};
