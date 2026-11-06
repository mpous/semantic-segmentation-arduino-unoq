# Segmentation Viewer

An Arduino App Lab app for the UNO Q. It runs the floor / obstacle segmentation model from the
[Edge Impulse custom learning block](https://github.com/mpous/semantic-segmentation-custom-learning-block)
on the USB camera and shows, in the browser:

- **left:** the camera image;
- **right:** the same image with the obstacle cells in red and the model's output grid;
- inference time, frame rate and the share of obstacle cells;
- **free floor near the robot:** the floor share in the left, centre and right thirds of the
  bottom 30% of the grid (below the white line on the mask). This is what an obstacle-avoidance
  loop would read;
- an **obstacle threshold** slider: a cell turns red once its obstacle probability reaches it.
  Lower values are more cautious.

The Edge Impulse deployment (`.eim`, object detection brick) merges the cells into boxes, so the
app reads the model's output tensor directly with LiteRT instead.

## Files

```
app.yaml                 web_ui brick
python/main.py           camera → model → two JPEGs → web UI
python/segmenter.py      LiteRT inference, mask drawing, free-space readout
python/requirements.txt  ai-edge-litert
assets/                  web page (index.html, app.js, style.css)
models/                  put the .tflite model here
```

## Set up

1. **Model.** In the Edge Impulse project, open the dashboard, download
   `model_quantized_int8_io.tflite` (the learning block's int8 output) and copy it to `models/`.
   `model.tflite` (float) also works; when there are several files, the int8 one is used.
2. **Copy the app to the board:**

   ```bash
   scp -r semantic-viewer-UNO-Q arduino@<board>.local:/home/arduino/ArduinoApps/
   ```

3. **Start it** from App Lab (*My Apps → Segmentation Viewer → Run*) or over SSH:

   ```bash
   arduino-app-cli app start user:semantic-viewer-UNO-Q
   arduino-app-cli app logs /home/arduino/ArduinoApps/semantic-viewer-UNO-Q --all
   ```

   The first start installs `ai-edge-litert` and takes a little longer. The log prints the model's
   input and output size once it's loaded.
4. **Open** `http://<board>.local:7000` on a computer on the same network.

## Notes

- The camera runs at 640×480, 10 fps. Each frame is squashed to the model input (320×320 for the
  current impulse), like the impulse's Squash resize, and converted from BGR to RGB (or grayscale
  for a grayscale impulse).
- Frames are only processed while a browser is connected.
- The *inference ms* figure is the LiteRT call alone; *fps* covers the whole loop, including
  JPEG encoding and sending both images.
- To change the camera resolution or frame rate, edit `Camera(...)` in `python/main.py`. The
  default threshold is in `settings`.
