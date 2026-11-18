'use strict';
const $ = id => document.getElementById(id);
const fmt = value => Number(value).toLocaleString('en-US');
const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const state = {trips: [], selected: null, result: null, busy: false, playback: null, controller: null, requestSerial: 0, connectionSerial: 0, health: null};
let apiBase = localStorage.getItem('pathsnap-api') || (location.port === '8001' ? 'http://127.0.0.1:8000' : location.origin);

function announce(text) {$('live-status').textContent = text;}
function showError(text) {$('error').textContent = text; $('error').hidden = !text; if (text) announce(text);}
function busy(value) {
  state.busy = value; $('busy').hidden = !value; $('run-button').disabled = value || !state.selected;
  $('run-button').querySelector('span').textContent = value ? 'Reconstructing…' : 'Reconstruct route';
}
async function request(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (!response.ok) {
    let detail = `API returned ${response.status}`;
    try {const body = await response.json(); if (typeof body.detail === 'string') detail = body.detail;} catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}
function renderTrips() {
  const filter = $('trip-search').value.trim().toLowerCase();
  const list = $('trip-list'); list.replaceChildren();
  const shown = state.trips.filter(t => t.trip_id.includes(filter) || t.label.toLowerCase().includes(filter));
  for (const trip of shown) {
    const button = document.createElement('button'); button.type = 'button';
    button.className = `trip-button${trip.trip_id === state.selected ? ' selected' : ''}`;
    button.setAttribute('aria-pressed', String(trip.trip_id === state.selected));
    const number = document.createElement('span'); number.className = 'trip-number'; number.textContent = Number(trip.trip_id).toString().padStart(2, '0');
    const text = document.createElement('span'); text.className = 'trip-text';
    const title = document.createElement('strong'); title.textContent = trip.label;
    const subtitle = document.createElement('span'); subtitle.textContent = `${fmt(trip.observations)} GPS observations`;
    text.append(title, subtitle);
    const arrow = document.createElement('span'); arrow.className = 'trip-arrow'; arrow.textContent = trip.trip_id === state.selected ? '↗' : '';
    button.append(number, text, arrow); button.addEventListener('click', () => selectTrip(trip.trip_id, true)); list.append(button);
  }
  if (!shown.length) {const empty = document.createElement('div'); empty.className = 'list-empty'; empty.textContent = state.trips.length ? 'No matching trips.' : 'No trips available. Check your dataset and API settings.'; list.append(empty);}
  const select = $('trip-select'); select.replaceChildren();
  for (const trip of state.trips) {const option = document.createElement('option'); option.value = trip.trip_id; option.textContent = `${trip.label} · ${fmt(trip.observations)} points`; option.selected = trip.trip_id === state.selected; select.append(option);}
}
function stopPlayback() {
  if (state.playback) clearInterval(state.playback);
  state.playback = null; $('play-button').innerHTML = '<svg><use href="#i-play"/></svg>'; $('play-button').setAttribute('aria-label', 'Play trajectory');
}
function clearResult() {
  state.result = null; map.setData(null); stopPlayback();
  for (const id of ['metric-accuracy','metric-continuity','metric-distance','metric-time','compare-classifier','compare-hmm','continuity-classifier','continuity-hmm','sequence-classifier','sequence-hmm','unmatched','resets','miss-rate','components']) $(id).textContent = '—';
  $('metric-accuracy-note').textContent = 'Waiting for a prediction'; $('metric-distance-note').textContent = 'Connected route pieces';
  $('label-notice-text').textContent = 'Metrics appear after reconstruction. Reference labels stay on the server.';
  $('observation-label').textContent = 'Observation —'; $('observation-time').textContent = '—'; $('timeline-total').textContent = '— points';
  $('export-button').disabled = true; $('timeline').disabled = true; $('play-button').disabled = true;
  $('map-chip').textContent = 'READY'; $('run-state').textContent = 'Ready when you are';
  $('run-info-text').textContent = 'Select a trip to inspect its reconstructed route.';
  $('map-title').textContent = 'Route preview'; $('map-subtitle').textContent = 'Select a trip to start exploring';
  $('map-scale').textContent = 'Vector map · no external tiles';
  $('break-notice').textContent = 'GPS gaps and disconnected roads are shown as route breaks.';
}
function selectTrip(id, run) {
  if (!state.trips.some(t => t.trip_id === id)) return;
  if (state.controller) state.controller.abort();
  state.requestSerial++; busy(false); state.selected = id; clearResult(); renderTrips(); showError('');
  const trip = state.trips.find(t => t.trip_id === id);
  $('map-title').textContent = `${trip.label} / Route reconstruction`;
  $('map-subtitle').textContent = `${id} · ${fmt(trip.observations)} GPS observations`;
  $('run-button').disabled = false;
  if (run) runPrediction();
}
async function connect(autoRun = true) {
  const connectionSerial = ++state.connectionSerial;
  $('connection').className = 'connection'; $('connection').innerHTML = '<i></i>Connecting';
  showError('');
  try {
    const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 8000);
    let health, data;
    try {[health, data] = await Promise.all([request('/health', {signal: controller.signal}), request('/trips', {signal: controller.signal})]);} finally {clearTimeout(timeout);}
    if (connectionSerial !== state.connectionSerial) return;
    state.health = health; state.trips = data.trips;
    $('connection').className = 'connection online'; $('connection').innerHTML = '<i></i>API online';
    $('trip-count').textContent = data.trips.length;
    $('partition-title').textContent = health.partition === 'evaluator_test' ? 'Evaluator testing' : health.partition === 'train' ? 'Training trips' : 'Student testing';
    $('footer-mode').textContent = `${health.mode} · server-side references`;
    $('api-docs').href = `${apiBase}/docs`;
    if (data.trips.length) selectTrip(state.trips.some(t => t.trip_id === state.selected) ? state.selected : data.trips[0].trip_id, autoRun);
    else {state.selected = null; clearResult(); renderTrips(); busy(false); if (data.notice) showError(data.notice);}
  } catch (error) {
    if (connectionSerial !== state.connectionSerial) return;
    $('connection').className = 'connection offline'; $('connection').innerHTML = '<i></i>API unavailable';
    state.trips = []; state.selected = null; clearResult(); renderTrips(); busy(false);
    $('trip-count').textContent = '0'; $('partition-title').textContent = 'Dataset unavailable'; $('footer-mode').textContent = 'API disconnected';
    showError(`Could not connect to ${apiBase}. Start the API on Laptop 1, then check the connection settings and network. ${error.name === 'AbortError' ? 'Connection timed out.' : error.message}`);
  }
}
async function runPrediction() {
  if (!state.selected || state.busy) return;
  const serial = ++state.requestSerial; const tripId = state.selected;
  state.controller = new AbortController(); stopPlayback(); busy(true); showError('');
  announce(`Reconstructing trip ${Number(tripId)}.`);
  const timeout = setTimeout(() => state.controller?.abort(), 180000);
  try {
    const result = await request('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({trip_id: tripId}), signal: state.controller.signal});
    if (serial !== state.requestSerial) return;
    state.result = result; showResult(result); announce(`Trip ${Number(tripId)} reconstructed. ${fmt(result.observation_count)} observations.`);
  } catch (error) {
    if (serial !== state.requestSerial) return;
    showError(error.name === 'AbortError' ? 'The prediction timed out. Check the server log and retry.' : `Reconstruction failed: ${error.message}`);
    $('map-chip').textContent = 'RETRY';
  } finally {clearTimeout(timeout); if (serial === state.requestSerial) busy(false);}
}
function showResult(data) {
  const evaluation = data.evaluation; const hs = data.hmm.summary;
  $('metric-accuracy').textContent = percent(evaluation.hmm.top1_accuracy);
  $('metric-accuracy-note').textContent = `${fmt(evaluation.eligible_observations)} of ${fmt(data.observation_count)} points eligible`;
  $('metric-continuity').textContent = percent(hs.continuity);
  $('metric-distance').innerHTML = `${(hs.route_length_m / 1000).toFixed(2)}<small>km</small>`;
  $('metric-distance-note').textContent = `${fmt(hs.route_pieces)} connected ${hs.route_pieces === 1 ? 'piece' : 'pieces'}`;
  $('metric-time').innerHTML = `${data.timings.total_prediction_s.toFixed(2)}<small>s</small>`;
  for (const name of ['classifier', 'hmm']) {
    $(`compare-${name}`).textContent = percent(evaluation[name].top1_accuracy);
    $(`continuity-${name}`).textContent = percent(data[name].summary.continuity);
    $(`sequence-${name}`).textContent = percent(evaluation[name].sequence_similarity);
  }
  $('unmatched').textContent = fmt(hs.unmatched_observations); $('resets').textContent = fmt(data.hmm.resets.length);
  $('miss-rate').textContent = percent(evaluation.candidate_miss_rate); $('components').textContent = fmt(data.network.weak_components);
  $('label-notice-text').textContent = evaluation.status === 'provisional_alignment' ? 'Provisional route alignment. Ambiguous points are excluded. These percentages are reference agreement, not verified GPS-point accuracy.' : evaluation.label_notice;
  const reasons = {}; for (const r of data.hmm.resets) reasons[r.reason] = (reasons[r.reason] || 0) + 1;
  const summaries = [];
  if (reasons.gps_gap) summaries.push(`${fmt(reasons.gps_gap)} large GPS ${reasons.gps_gap === 1 ? 'gap' : 'gaps'}`);
  if (reasons.infeasible_transition) summaries.push(`${fmt(reasons.infeasible_transition)} disconnected ${reasons.infeasible_transition === 1 ? 'transition' : 'transitions'}`);
  if (reasons.no_candidates) summaries.push(`${fmt(reasons.no_candidates)} observations without candidates`);
  $('break-notice').textContent = summaries.length ? `Route breaks preserve ${summaries.join(' and ')}.` : 'No decoder restarts were needed. The reconstructed positions form a connected sequence.';
  $('map-chip').textContent = 'RECONSTRUCTED'; $('run-state').textContent = 'Live run complete';
  $('run-info-text').textContent = `Request ${data.request_id} · Model ${data.model_version} · ${data.timings.request_s.toFixed(2)} s total`;
  $('timeline').max = data.observation_count - 1; $('timeline').value = 0; $('timeline').disabled = false;
  $('timeline-total').textContent = `${fmt(data.observation_count)} points`; $('export-button').disabled = false; $('play-button').disabled = false;
  map.setData(data); updateObservation(0);
}
function updateObservation(index) {
  const data = state.result; if (!data) return;
  index = Math.max(0, Math.min(data.observation_count - 1, Number(index)));
  $('timeline').value = index; $('observation-label').textContent = `Observation ${fmt(index + 1)} / ${fmt(data.observation_count)}`;
  const elapsed = data.timestamps_s[index] - data.timestamps_s[0];
  $('observation-time').textContent = `+${fmt(Math.round(elapsed))} s`;
  map.observation = index; map.draw();
}

class RouteMap {
  constructor(canvas) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.data = null;
    this.scale = 1; this.center = [0, 0]; this.observation = 0; this.drag = null;
    this.layers = {roads: true, gps: true, classifier: false, hmm: true};
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.addEventListener('wheel', e => {e.preventDefault(); const rect = canvas.getBoundingClientRect(); this.zoom(Math.exp(-e.deltaY * .0015), e.clientX - rect.left, e.clientY - rect.top);}, {passive: false});
    canvas.addEventListener('pointerdown', e => {this.drag = {x: e.clientX, y: e.clientY, center: [...this.center], moved: false}; canvas.setPointerCapture(e.pointerId); canvas.classList.add('dragging'); $('map-tooltip').hidden = true;});
    canvas.addEventListener('pointermove', e => {
      if (this.drag) {const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y; this.center = [this.drag.center[0] - dx / this.scale, this.drag.center[1] - dy / this.scale]; this.drag.moved ||= Math.abs(dx) + Math.abs(dy) > 5; this.draw();}
      else this.hover(e);
    });
    canvas.addEventListener('pointerup', e => {const wasDrag = this.drag?.moved; this.drag = null; canvas.classList.remove('dragging'); if (!wasDrag) this.hover(e, true);});
    canvas.addEventListener('pointercancel', () => {this.drag = null; canvas.classList.remove('dragging');});
    canvas.addEventListener('pointerleave', () => {$('map-tooltip').hidden = true;});
    canvas.addEventListener('keydown', e => {
      if (['+', '=', '-'].includes(e.key)) {e.preventDefault(); this.zoom(e.key === '-' ? .8 : 1.25);}
      if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) {e.preventDefault(); this.center[0] += (e.key === 'ArrowRight' ? 60 : e.key === 'ArrowLeft' ? -60 : 0) / this.scale; this.center[1] += (e.key === 'ArrowDown' ? 60 : e.key === 'ArrowUp' ? -60 : 0) / this.scale; this.draw();}
    });
  }
  resize() {
    const rect = this.canvas.getBoundingClientRect(); this.w = rect.width; this.h = rect.height;
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.w * ratio); this.canvas.height = Math.round(this.h * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (this.data) this.fit(); else this.draw();
  }
  local(point) {return [(point[0] - this.origin[0]) * this.lonScale, -(point[1] - this.origin[1]) * 111195.08];}
  pixel(point) {return [(point[0] - this.center[0]) * this.scale + this.w / 2, (point[1] - this.center[1]) * this.scale + this.h / 2];}
  setData(data) {
    this.data = data; $('map-empty').hidden = Boolean(data); $('map-tooltip').hidden = true;
    if (!data) {this.draw(); return;}
    this.origin = data.gps.reduce((a, p) => [a[0] + p[0] / data.gps.length, a[1] + p[1] / data.gps.length], [0,0]);
    this.lonScale = 111195.08 * Math.cos(this.origin[1] * Math.PI / 180);
    this.gps = data.gps.map(p => this.local(p)); this.roads = data.road_segments.map(line => line.map(p => this.local(p)));
    this.classifier = data.classifier.pieces.map(piece => piece.map(p => this.local(p)));
    this.hmm = data.hmm.pieces.map(piece => piece.map(p => this.local(p)));
    this.snapped = data.hmm.snapped.map(p => p ? this.local(p) : null); this.observation = 0; this.fit();
  }
  fit() {
    if (!this.data) return;
    const points = [...this.gps, ...this.hmm.flat()];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of points) {minX = Math.min(minX,p[0]); maxX = Math.max(maxX,p[0]); minY = Math.min(minY,p[1]); maxY = Math.max(maxY,p[1]);}
    this.center = [(minX + maxX) / 2, (minY + maxY) / 2];
    this.scale = Math.min((this.w - 100) / Math.max(maxX - minX, 50), (this.h - 120) / Math.max(maxY - minY, 50)); this.draw();
  }
  zoom(factor, x = this.w / 2, y = this.h / 2) {
    if (!this.data) return;
    const old = this.scale, next = Math.max(.0005, Math.min(40, old * factor));
    this.center[0] += (x - this.w / 2) * (1 / old - 1 / next); this.center[1] += (y - this.h / 2) * (1 / old - 1 / next); this.scale = next; this.draw();
  }
  lines(pieces, color, width, dash = []) {
    const c = this.ctx; c.beginPath(); c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash);
    for (const piece of pieces) {if (piece.length < 2) continue; let p = this.pixel(piece[0]); c.moveTo(...p); for (let i = 1; i < piece.length; i++) c.lineTo(...this.pixel(piece[i]));}
    c.stroke(); c.setLineDash([]);
  }
  dot(point, radius, fill, border = '#fff') {
    const c = this.ctx, p = this.pixel(point); c.beginPath(); c.arc(...p, radius, 0, Math.PI * 2); c.fillStyle = fill; c.fill(); c.strokeStyle = border; c.lineWidth = 2; c.stroke();
  }
  draw() {
    const c = this.ctx; if (!c || !this.w) return;
    c.clearRect(0, 0, this.w, this.h); c.fillStyle = '#edf0e9'; c.fillRect(0, 0, this.w, this.h);
    const grid = 80; c.beginPath(); c.strokeStyle = '#e5eadd'; c.lineWidth = .6;
    for (let x=0;x<this.w;x+=grid) {c.moveTo(x,0);c.lineTo(x,this.h);} for(let y=0;y<this.h;y+=grid){c.moveTo(0,y);c.lineTo(this.w,y);} c.stroke();
    if (!this.data) return;
    c.lineCap = 'round'; c.lineJoin = 'round';
    if (this.layers.roads) {this.lines(this.roads, '#fffdf8', 4); this.lines(this.roads, '#d0d8c8', .8);}
    if (this.layers.gps) {
      this.lines([this.gps], '#5e86b07a', 1.2, [3,5]);
      const step = Math.max(1, Math.ceil(this.gps.length / 600)); c.fillStyle = '#6586ac';
      for (let i=0; i<this.gps.length; i+=step) {const p=this.pixel(this.gps[i]); c.beginPath();c.arc(...p,1.6,0,Math.PI*2);c.fill();}
    }
    if (this.layers.classifier) {this.lines(this.classifier, '#fff8e9', 6); this.lines(this.classifier, '#c9913c', 2.6);}
    if (this.layers.hmm) {this.lines(this.hmm, '#ffffffcc', 5.5); this.lines(this.hmm, '#087f72', 2.8);}
    this.dot(this.gps[0], 6, '#087c70'); this.dot(this.gps.at(-1), 6, '#b86d48');
    const active = this.gps[this.observation], snap = this.snapped[this.observation];
    if (snap) {this.lines([[active,snap]], '#42867570', 1.2, [3,3]); this.dot(snap,4,'#087c70');}
    this.dot(active, 7, '#5c83b0'); this.dot(active, 2, '#fff', '#fff');
    const nominal = 90 / this.scale; const power = 10 ** Math.floor(Math.log10(nominal)); const length = [1,2,5,10].map(n=>n*power).filter(n=>n<=nominal).at(-1) || nominal;
    const unit = length >= 1000 ? `${(length/1000).toFixed(length % 1000 ? 1 : 0)} km` : `${Math.round(length)} m`;
    const bar = document.createElement('span'); bar.textContent = unit; bar.style.cssText = `display:inline-block;width:${Math.round(length*this.scale)}px;border-bottom:1px solid #7d8a77;text-align:center;margin-right:8px;padding-bottom:2px`;
    const label = document.createElement('span'); label.textContent = 'Vector roads · no external tiles'; $('map-scale').replaceChildren(bar,label);
  }
  hover(event, pick = false) {
    if (!this.data) return;
    const rect = this.canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    let best = -1, distance = 144;
    for (let i=0; i<this.gps.length; i++) {const p=this.pixel(this.gps[i]), d=(p[0]-x)**2+(p[1]-y)**2; if(d<distance){distance=d;best=i;}}
    const tip = $('map-tooltip'); tip.hidden = best < 0;
    if (best >= 0) {
      const arc = this.data.hmm.assignments[best]; tip.replaceChildren();
      const title = document.createElement('strong'); title.textContent = `Observation ${fmt(best+1)}`;
      const line = document.createElement('div'); line.textContent = arc >= 0 ? `Matched arc ${arc}` : 'No road candidate'; tip.append(title,line);
      tip.style.left = `${Math.min(x + 14, this.w - 160)}px`; tip.style.top = `${Math.max(8,y - 50)}px`;
      if (pick) {stopPlayback(); updateObservation(best);}
    }
  }
}
const map = new RouteMap($('map'));
$('trip-search').addEventListener('input', renderTrips);
$('trip-select').addEventListener('change', e => selectTrip(e.target.value, true));
$('run-button').addEventListener('click', runPrediction);
$('zoom-in').addEventListener('click', () => map.zoom(1.35)); $('zoom-out').addEventListener('click', () => map.zoom(1/1.35)); $('fit-button').addEventListener('click', () => map.fit());
for (const layer of ['roads','gps','classifier','hmm']) $(`layer-${layer}`).addEventListener('change', e => {map.layers[layer] = e.target.checked; map.draw();});
$('timeline').addEventListener('input', e => {stopPlayback(); updateObservation(e.target.value);});
$('play-button').addEventListener('click', () => {
  if (!state.result) return; if (state.playback) {stopPlayback();return;}
  if (Number($('timeline').value) === state.result.observation_count-1) updateObservation(0);
  $('play-button').textContent = 'Ⅱ'; $('play-button').setAttribute('aria-label','Pause trajectory');
  state.playback = setInterval(() => {const next=Math.min(state.result.observation_count-1,Number($('timeline').value)+Math.max(1,Math.ceil(state.result.observation_count/350)));updateObservation(next);if(next===state.result.observation_count-1)stopPlayback();},70);
});
$('settings-button').addEventListener('click', () => {$('api-url').value=apiBase;$('settings-dialog').showModal();});
$('settings-cancel').addEventListener('click', () => $('settings-dialog').close());
$('settings-form').addEventListener('submit', e => {
  e.preventDefault(); const input=$('api-url'); let url;
  try {url=new URL(input.value); if(!['http:','https:'].includes(url.protocol))throw new Error();} catch(_){input.setCustomValidity('Enter an http:// or https:// API address.');input.reportValidity();return;}
  input.setCustomValidity(''); state.controller?.abort(); state.requestSerial++; busy(false);
  apiBase=url.origin; localStorage.setItem('pathsnap-api',apiBase); $('settings-dialog').close(); connect(true);
});
$('api-url').addEventListener('input', e => e.target.setCustomValidity(''));
$('how-button').addEventListener('click', () => $('how-dialog').showModal()); $('how-close').addEventListener('click', () => $('how-dialog').close());
$('export-button').addEventListener('click', () => {if(!state.result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(state.result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`pathsnap_${state.result.trip_id}_${state.result.request_id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
connect(true);
