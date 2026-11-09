"""Upload images and see the segmentation model's mask, without a camera or a board.

Runs the same code as the App Lab app (python/segmenter.py), so the result is
what the UNO Q would show for that frame. Works on a laptop or on the board:

    python3 -m venv ~/.venvs/seg-upload && ~/.venvs/seg-upload/bin/pip install -r upload/requirements.txt
    ~/.venvs/seg-upload/bin/python upload/app.py        # model from models/
    ~/.venvs/seg-upload/bin/python upload/app.py --model path/to/model_quantized_int8_io.tflite

then open http://127.0.0.1:5001. Class names come from metrics.json (or
labels.txt) next to the model. Images are processed in memory, nothing is saved.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from segmenter import Segmenter, class_map, coverage, draw_mask, load_classes  # noqa: E402

MAX_FILES = 20
MAX_SIDE = 800  # longest side of the images sent back
JPEG_QUALITY = 85


def find_model(path: Path | None) -> Path:
    if path is not None:
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        return path
    models_dir = ROOT / "models"
    preferred = models_dir / "model_quantized_int8_io.tflite"
    if preferred.exists():
        return preferred
    models = sorted(models_dir.glob("*.tflite"))
    if not models:
        raise SystemExit(f"No .tflite model in {models_dir}. Copy model_quantized_int8_io.tflite there, or pass --model.")
    return models[0]


def jpeg_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def shrink(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def create_app(model_path: Path) -> Flask:
    seg = Segmenter(model_path)
    classes = load_classes(model_path.parent, seg.classes)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    info = {
        "model": model_path.name, "input": f"{seg.width}x{seg.height}x{seg.channels}",
        "grid": f"{seg.cols}x{seg.rows}", "background": classes.names[0],
        "classes": classes.legend(), "names_from": classes.source,
    }

    @app.get("/")
    def index():
        return render_template_string(PAGE, info=info)

    @app.post("/predict")
    def predict():
        try:
            threshold = min(max(float(request.form.get("threshold", 0.5)), 0.05), 0.95)
        except ValueError:
            return jsonify(error="threshold must be a number"), 400
        files = request.files.getlist("images")
        if not files:
            return jsonify(error="no images"), 400
        if len(files) > MAX_FILES:
            return jsonify(error=f"at most {MAX_FILES} images at a time"), 400

        results = []
        for f in files:
            name = Path(f.filename or "image").name
            frame = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                results.append({"name": name, "error": "not an image OpenCV can read"})
                continue
            start = time.perf_counter()
            probs = seg.probabilities(frame)
            inference_ms = (time.perf_counter() - start) * 1000
            cmap = class_map(probs, classes, threshold)
            shown = shrink(frame)
            h, w = frame.shape[:2]
            results.append({
                "name": name,
                "size": f"{w}x{h}",
                "image": jpeg_data_url(shown),
                "mask": jpeg_data_url(draw_mask(shown, cmap, classes, near_line=False)),
                "inference_ms": round(inference_ms, 1),
                "coverage": {k: round(v, 4) for k, v in coverage(cmap, classes).items()},
            })
        return jsonify(threshold=threshold, results=results)

    return app


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Segmentation: upload images</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background: #f4f6f7; color: #1f2a30; }
  header { padding: 16px 24px; background: #008184; color: #fff; }
  h1 { margin: 0; font-size: 20px; font-weight: 600; }
  header p { margin: 4px 0 0; font-size: 13px; opacity: 0.85; }
  main { padding: 24px; max-width: 1400px; margin: 0 auto; }
  .card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 16px; }
  form { display: grid; grid-template-columns: 1fr 260px auto; gap: 16px; align-items: end; }
  label { display: block; font-size: 14px; font-weight: 600; margin-bottom: 6px; }
  input[type=range] { width: 100%; accent-color: #008184; }
  button { background: #008184; color: #fff; border: 0; border-radius: 6px; padding: 9px 18px; font-size: 14px; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  small { color: #5f6b70; font-size: 12px; }
  .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; margin-top: 12px; }
  .legend i, .cov i { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; opacity: .7; border-radius: 2px; }
  .result h2 { margin: 0 0 8px; font-size: 15px; display: flex; justify-content: space-between; gap: 12px; }
  .result h2 small { font-weight: 400; }
  .views { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
  .views img { width: 100%; display: block; border-radius: 4px; background: #111; }
  .covs { margin-top: 12px; display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 4px 24px; }
  .cov { display: grid; grid-template-columns: minmax(80px, auto) 1fr 48px; gap: 8px; align-items: center; font-size: 13px; }
  .bar { height: 10px; background: #e3e7e9; border-radius: 5px; overflow: hidden; }
  .bar > div { height: 100%; }
  .cov b { text-align: right; font-weight: 600; }
  .error { color: #b3261e; font-size: 14px; }
</style>
</head>
<body>
<header>
  <h1>Segmentation: upload images</h1>
  <p>{{ info.model }} · input {{ info.input }} · grid {{ info.grid }} · class names from {{ info.names_from }}</p>
</header>
<main>
  <div class="card">
    <form id="form">
      <div>
        <label for="images">Images</label>
        <input id="images" name="images" type="file" accept="image/*" multiple required />
        <div><small>Each image is squashed to the model input, like the impulse. Up to 20 at a time.</small></div>
      </div>
      <div>
        <label for="threshold">Class threshold <b id="threshold-v">0.50</b></label>
        <input id="threshold" name="threshold" type="range" min="0.05" max="0.95" step="0.05" value="0.5" />
        <small>Lower marks cells as a class sooner.</small>
      </div>
      <button id="run" type="submit">Run model</button>
    </form>
    <div class="legend" id="legend"></div>
    <p class="error" id="error" hidden></p>
  </div>
  <div id="results"></div>
</main>
<script>
const INFO = {{ info | tojson }};
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const swatch = (colour) => { const i = el('i'); i.style.background = colour; return i; };
const percent = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
const colours = { [INFO.background]: '#c3cbcf', ...Object.fromEntries(INFO.classes.map((c) => [c.name, c.colour])) };

$('legend').append(...INFO.classes.map((c) => { const s = el('span'); s.append(swatch(c.colour), c.name); return s; }));

function render(data) {
  $('results').replaceChildren(...data.results.map((r) => {
    const card = el('div', 'card result');
    const title = el('h2', null, r.name);
    if (r.error) { card.append(title, el('p', 'error', r.error)); return card; }
    title.append(el('small', null, `${r.size} · ${r.inference_ms.toFixed(r.inference_ms < 10 ? 1 : 0)} ms`));
    const views = el('div', 'views');
    for (const [src, alt] of [[r.image, 'Image'], [r.mask, 'Predicted classes']]) {
      const img = el('img'); img.src = src; img.alt = `${alt}: ${r.name}`; views.append(img);
    }
    const covs = el('div', 'covs');
    for (const [name, v] of Object.entries(r.coverage)) {
      const row = el('div', 'cov'), label = el('span'), bar = el('div', 'bar'), fill = el('div');
      label.append(swatch(colours[name]), name);
      fill.style.width = percent(v); fill.style.background = colours[name];
      bar.append(fill);
      row.append(label, bar, el('b', null, percent(v)));
      covs.append(row);
    }
    card.append(title, views, covs);
    return card;
  }));
}

async function run() {
  if (!$('images').files.length) return;
  $('run').disabled = true; $('error').hidden = true;
  try {
    const res = await fetch('predict', { method: 'POST', body: new FormData($('form')) });
    const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    render(data);
  } catch (e) {
    $('error').textContent = `Failed: ${e.message}`; $('error').hidden = false;
  } finally {
    $('run').disabled = false;
  }
}

$('form').addEventListener('submit', (e) => { e.preventDefault(); run(); });
$('threshold').addEventListener('input', () => { $('threshold-v').textContent = Number($('threshold').value).toFixed(2); });
$('threshold').addEventListener('change', run);  // re-run the same images at the new threshold
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Upload images and see the segmentation model's mask.")
    p.add_argument("--model", type=Path, default=None, help=".tflite from the learning block (default: models/)")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines (e.g. on the UNO Q)")
    p.add_argument("--port", type=int, default=5001)
    args = p.parse_args(argv)

    model_path = find_model(args.model)
    app = create_app(model_path)
    print(f"Open http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
