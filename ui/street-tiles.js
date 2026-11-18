'use strict';

// Web Mercator coordinates, with a top-left origin and y increasing south.
const WORLD_SIZE = 2 * Math.PI * 6378137;
const TILE_SIZE = 256;
const MAX_TILE_ZOOM = 19;
const STREET_TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

function mapCoordinate([longitude, latitude]) {
  const phi = Math.max(-85.05112878, Math.min(85.05112878, latitude)) * Math.PI / 180;
  return [(longitude + 180) * WORLD_SIZE / 360,
    WORLD_SIZE * (.5 - Math.asinh(Math.tan(phi)) / (2 * Math.PI))];
}

class StreetTiles {
  constructor(container, onChange) {
    this.container = container;
    this.onChange = onChange;
    this.cache = new Map();
    this.visible = new Map();
  }

  clear() {
    this.visible.clear();
    this.container.replaceChildren();
    this.container.hidden = true;
  }

  tile(key, zoom, x, y) {
    let tile = this.cache.get(key);
    if (tile) {
      this.cache.delete(key);
      this.cache.set(key, tile);
      return tile;
    }
    const image = new Image();
    image.alt = '';
    image.draggable = false;
    image.decoding = 'async';
    image.referrerPolicy = 'strict-origin-when-cross-origin';
    tile = {image, status: 'loading'};
    const finish = status => {
      tile.status = status;
      if ([...this.visible.values()].some(entry => entry.tile === tile)) this.onChange();
    };
    image.onload = () => finish('loaded');
    image.onerror = () => finish('failed');
    this.cache.set(key, tile);
    image.src = STREET_TILE_URL.replace('{z}', zoom).replace('{x}', x).replace('{y}', y);
    return tile;
  }

  render(center, scale, width, height) {
    const zoom = Math.max(0, Math.min(MAX_TILE_ZOOM, Math.round(Math.log2(scale * WORLD_SIZE / TILE_SIZE))));
    const count = 2 ** zoom;
    const span = WORLD_SIZE / count;
    const left = center[0] - width / (2 * scale);
    const top = center[1] - height / (2 * scale);
    const maxX = Math.ceil((left + width / scale) / span) - 1;
    const maxY = Math.min(count - 1, Math.ceil((top + height / scale) / span) - 1);
    const next = new Map();
    for (let y = Math.max(0, Math.floor(top / span)); y <= maxY; y++) {
      for (let x = Math.floor(left / span); x <= maxX; x++) {
        const wrappedX = ((x % count) + count) % count;
        const key = `${zoom}/${wrappedX}/${y}`;
        const position = `${zoom}/${x}/${y}`;
        const tile = this.tile(key, zoom, wrappedX, y);
        // The same image can appear more than once near the world seam.
        const image = this.visible.get(position)?.image ||
          (tile.image.parentNode ? tile.image.cloneNode() : tile.image);
        const px = Math.round((x * span - left) * scale);
        const py = Math.round((y * span - top) * scale);
        image.style.transform = `translate(${px}px, ${py}px)`;
        image.style.width = `${Math.round(((x + 1) * span - left) * scale) - px}px`;
        image.style.height = `${Math.round(((y + 1) * span - top) * scale) - py}px`;
        if (image.parentNode !== this.container) this.container.append(image);
        next.set(position, {tile, image});
      }
    }
    for (const [position, entry] of this.visible) {
      if (!next.has(position)) entry.image.remove();
    }
    this.visible = next;
    this.container.hidden = false;
    const active = new Set([...next.values()].map(entry => entry.tile));
    for (const [key, tile] of this.cache) {
      if (this.cache.size <= 96) break;
      if (!active.has(tile)) this.cache.delete(key);
    }
    const tiles = [...active];
    return {loaded: tiles.some(tile => tile.status === 'loaded'),
      loading: tiles.some(tile => tile.status === 'loading')};
  }
}
