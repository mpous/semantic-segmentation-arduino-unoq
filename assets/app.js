const ui = new WebUI();
const $ = (id) => document.getElementById(id);
const percent = (v) => `${Math.round(v * 100)}%`;

const status = $('status');
const slider = $('threshold');
let classes = [];
let background = 'background';

const swatch = (colour) => {
  const i = document.createElement('i');
  i.style.background = colour;
  return i;
};

// One legend entry and one coverage row per class (the background has no colour).
function setClasses(list, bg) {
  classes = list;
  background = bg;
  $('background').textContent = bg;
  $('legend').replaceChildren(...list.map((c) => {
    const span = document.createElement('span');
    span.append(swatch(c.colour), c.name);
    return span;
  }));
  $('coverage').replaceChildren(...[{ name: bg, colour: '#c3cbcf' }, ...list].map((c) => {
    const row = document.createElement('div');
    row.className = 'cov';
    const name = document.createElement('span');
    name.append(swatch(c.colour), c.name);
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('div');
    fill.style.background = c.colour;
    fill.dataset.name = c.name;
    bar.append(fill);
    const value = document.createElement('b');
    value.dataset.name = c.name;
    value.textContent = '–';
    row.append(name, bar, value);
    return row;
  }));
}

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
  $('model').textContent = `${s.model} · input ${s.input} · grid ${s.grid} · class names from ${s.names_from}`;
  setClasses(s.classes, s.background);
});

ui.on_message('frame', (f) => {
  $('camera').src = `data:image/jpeg;base64,${f.camera}`;
  $('mask').src = `data:image/jpeg;base64,${f.mask}`;
  $('inference').textContent = f.inference_ms.toFixed(0);
  $('fps').textContent = f.fps ? f.fps.toFixed(1) : '–';
  $('objects').textContent = percent(1 - (f.coverage[background] ?? 1));
  $('coverage').querySelectorAll('.bar > div').forEach((el) => {
    el.style.width = percent(f.coverage[el.dataset.name] ?? 0);
  });
  $('coverage').querySelectorAll('b').forEach((el) => {
    el.textContent = percent(f.coverage[el.dataset.name] ?? 0);
  });
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
