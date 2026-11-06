const ui = new WebUI();
const $ = (id) => document.getElementById(id);
const percent = (v) => `${Math.round(v * 100)}%`;

const status = $('status');
const slider = $('threshold');

ui.on_connect(() => {
  status.textContent = 'Live';
  status.className = 'status online';
});

ui.on_disconnect(() => {
  status.textContent = 'Disconnected';
  status.className = 'status offline';
});

ui.on_message('settings', (s) => {
  slider.value = s.threshold;
  $('threshold-v').textContent = Number(s.threshold).toFixed(2);
  $('model').textContent = `${s.model} · input ${s.input} · grid ${s.grid}`;
});

ui.on_message('frame', (f) => {
  $('camera').src = `data:image/jpeg;base64,${f.camera}`;
  $('mask').src = `data:image/jpeg;base64,${f.mask}`;
  $('inference').textContent = f.inference_ms.toFixed(0);
  $('fps').textContent = f.fps ? f.fps.toFixed(1) : '–';
  $('obstacle').textContent = percent(f.obstacle);
  f.free.forEach((v, i) => {
    const bar = $(`free-${i}`);
    bar.style.width = percent(v);
    bar.className = v >= 0.8 ? 'ok' : v >= 0.5 ? 'warn' : 'bad';
    $(`free-${i}-v`).textContent = percent(v);
  });
});

slider.addEventListener('input', () => {
  $('threshold-v').textContent = Number(slider.value).toFixed(2);
});

slider.addEventListener('change', () => {
  ui.send_message('set_threshold', { threshold: Number(slider.value) });
});
