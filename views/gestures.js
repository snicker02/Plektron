// Shared pointer handling: one-finger drag, pinch zoom, two-finger pan, tap, double-tap, wheel.
export function attachGestures(el, h) {
  const pts = new Map();
  let start = null, moved = false, lastTap = 0, lastTapXY = null, tapHandledAt = 0;
  let pinch = null;
  const pos = (e) => { const r = el.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  el.addEventListener('pointerdown', (e) => {
    el.setPointerCapture && el.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, pos(e));
    if (pts.size === 1) { start = { ...pos(e), t: performance.now() }; moved = false; h.dragStart && h.dragStart(start.x, start.y); }
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 };
      moved = true;
    }
  });
  el.addEventListener('pointermove', (e) => {
    if (!pts.has(e.pointerId)) { h.hover && h.hover(pos(e).x, pos(e).y); return; }
    const p = pos(e), prev = pts.get(e.pointerId); pts.set(e.pointerId, p);
    if (pts.size === 1 && start) {
      if (Math.hypot(p.x - start.x, p.y - start.y) > 6) moved = true;
      if (moved) h.drag && h.drag(p.x - prev.x, p.y - prev.y, p.x, p.y);
    } else if (pts.size === 2 && pinch) {
      const [a, b] = [...pts.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y) || 1, cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
      h.zoom && h.zoom(d / pinch.d, cx, cy);
      h.pan && h.pan(cx - pinch.cx, cy - pinch.cy);
      pinch = { d, cx, cy };
    }
  });
  const end = (e, cancelled) => {
    if (!pts.has(e.pointerId)) return;
    pts.delete(e.pointerId);
    if (pts.size < 2) pinch = null;
    if (pts.size === 0) {
      if (!cancelled && !moved && start && performance.now() - start.t < 500) {
        const now = performance.now();
        const isDouble = now - lastTap < 320 && lastTapXY && Math.hypot(lastTapXY.x - start.x, lastTapXY.y - start.y) < 24 && now - tapHandledAt > 400;
        if (isDouble) { tapHandledAt = now; lastTap = 0; h.doubleTap && h.doubleTap(start.x, start.y); }
        else { lastTap = now; lastTapXY = { x: start.x, y: start.y }; h.tap && h.tap(start.x, start.y); }
      }
      h.dragEnd && h.dragEnd();
      start = null;
    }
  };
  el.addEventListener('pointerup', (e) => end(e, false));
  el.addEventListener('pointercancel', (e) => end(e, true));
  el.addEventListener('dblclick', (e) => e.preventDefault());
  el.addEventListener('wheel', (e) => {
    e.preventDefault();
    const p = pos(e);
    if (e.ctrlKey || e.metaKey) h.zoom && h.zoom(Math.exp(-e.deltaY * 0.01), p.x, p.y);
    else h.wheel && h.wheel(e.deltaX, e.deltaY, e.shiftKey);
  }, { passive: false });
}

/** Keep a canvas backing store matched to its own CSS box. */
export function fitCanvas(cv) {
  const r = cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  return { w: r.width, h: r.height, dpr };
}

export const STRING_COLORS = ['#e5604f', '#f2a23a', '#e8d44d', '#6cc96a', '#4fc3c9', '#8a8cf0', '#d07de0'];
