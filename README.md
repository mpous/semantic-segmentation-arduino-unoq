# Segmentation Viewer

An Arduino App Lab app for the UNO Q. It runs a semantic segmentation model from the
[Edge Impulse custom learning block](https://github.com/mpous/semantic-segmentation-custom-learning-block)
on the USB camera, with any classes (floor / obstacle, ball / referee / teamA / teamB, …), and
shows, in the browser:

- **left:** the camera image;
- **right:** the same image with each class's cells in its own colour (legend above it) and the
  model's output grid. The background (the labeler's default class) is left clear;
- inference time, frame rate, and the share of the image each class covers;
- **free background near the camera:** the background share in the left, centre and right thirds
  of the bottom 30% of the grid (below the white line on the mask). For a robot, with `floor` as
  the background, this is what an obstacle-avoidance loop would read;
- a **class threshold** slider: a cell gets a class once all the classes but the background
  together reach this probability (the most likely of them wins). Lower values are more cautious.

There is also an [upload app](#upload-app-no-camera-or-board-needed) to try the model on image
files from a laptop.

The Edge Impulse deployment (`.eim`, object detection brick) merges the cells into boxes, so the
app reads the model's output tensor directly with LiteRT instead.

## Files

```
app.yaml                 web_ui brick
python/main.py           camera → model → two JPEGs → web UI
python/segmenter.py      LiteRT inference, class names, mask drawing, readouts
python/requirements.txt  ai-edge-litert
assets/                  web page (index.html, app.js, style.css)
models/                  put the .tflite model (and metrics.json) here
upload/app.py            upload app: run the model on image files in the browser
```

## Set up

1. **Model.** In the Edge Impulse project, open the dashboard, download
   `model_quantized_int8_io.tflite` (the learning block's int8 output) and copy it to `models/`.
   `model.tflite` (float) also works; when there are several files, the int8 one is used.
2. **Class names.** The `.tflite` file only has numbered outputs. Download `metrics.json` from
   the training job (the learning block's custom data) and copy it to `models/` too: the app takes
   the class names from it and hides the untrained `ignore` channel. Instead, you can write a
   `models/labels.txt` with one name per line, background first, then the Studio labels in
   alphabetical order (the order on the extension's *Learning block settings* card), e.g.
   `floor`, `ignore`, `obstacle`. Without either, the classes show as *class 1*, *class 2*, ….
3. **Copy the app to the board:**

   ```bash
   scp -r semantic-viewer-UNO-Q arduino@<board>.local:/home/arduino/ArduinoApps/
   ```

4. **Start it** from App Lab (*My Apps → Segmentation Viewer → Run*) or over SSH:

   ```bash
   arduino-app-cli app start user:semantic-viewer-UNO-Q
   arduino-app-cli app logs /home/arduino/ArduinoApps/semantic-viewer-UNO-Q --all
   ```

   The first start installs `ai-edge-litert` and takes a little longer. The log prints the model's
   input and output size and the class names once it's loaded.
5. **Open** `http://<board>.local:7000` on a computer on the same network.

## Notes

- The camera runs at 640×480, 10 fps. Each frame is squashed to the model input (320×320 for the
  current impulse), like the impulse's Squash resize, and converted from BGR to RGB (or grayscale
  for a grayscale impulse).
- Frames are only processed while a browser is connected.
- The *inference ms* figure is the LiteRT call alone; *fps* covers the whole loop, including
  JPEG encoding and sending both images.
- To change the camera resolution or frame rate, edit `Camera(...)` in `python/main.py`. The
  default threshold is in `settings`.

## Upload app (no camera or board needed)

`upload/app.py` is a small Flask app: pick one or more images in the browser and see the same
mask, class shares and inference time the UNO Q would show for them. Changing the threshold
re-runs the same images. It uses `python/segmenter.py`, so the results match the App Lab app.
Images are processed in memory and nothing is saved.

```bash
cd semantic-viewer-UNO-Q
# outside the app folder, so it isn't copied to the board with the app
python3 -m venv ~/.venvs/seg-upload && ~/.venvs/seg-upload/bin/pip install -r upload/requirements.txt
~/.venvs/seg-upload/bin/python upload/app.py              # uses models/, like the App Lab app
~/.venvs/seg-upload/bin/python upload/app.py --model ~/Downloads/model_quantized_int8_io.tflite
```

Open http://127.0.0.1:5001. The class names come from `metrics.json` or `labels.txt` next to the
model. The app listens on this computer only; add `--host 0.0.0.0` to reach it from another
machine (for example when running it on the UNO Q itself, while the App Lab app is stopped).

On macOS with Python 3.9, recent `ai-edge-litert` wheels fail to load
(`Library not loaded: @rpath/libpywrap_litert_common.dylib`). Install
`pip install "ai-edge-litert==1.2.0"`, or TensorFlow: the segmenter falls back to
`tensorflow.lite` when LiteRT can't be imported.
