(function () {
  var SVG_NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, children) {
    var n = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (var k in attrs) n.setAttribute(k, attrs[k]);
    }
    if (children) {
      children.forEach(function (c) { n.appendChild(c); });
    }
    return n;
  }

  function txt(tag, attrs, content) {
    var n = el(tag, attrs);
    n.textContent = content;
    return n;
  }

  function renderResolutionScatter(container, data) {
    container.innerHTML = '';
    if (!data || !data.length) return;

    var totalAll = 0;
    data.forEach(function (d) { totalAll += d.count; });
    var minCount = Math.max(2, Math.ceil(totalAll * 0.01));
    var filtered = data.filter(function (d) { return d.count >= minCount; });
    if (!filtered.length) filtered = data;
    data = filtered;

    var W = 360, H = 220;
    var pad = { l: 44, r: 14, t: 12, b: 34 };
    var maxW = 0, maxH = 0, maxC = 0;
    data.forEach(function (d) {
      if (d.w > maxW) maxW = d.w;
      if (d.h > maxH) maxH = d.h;
      if (d.count > maxC) maxC = d.count;
    });
    var xMax = Math.ceil(maxW / 100) * 100;
    var yMax = Math.ceil(maxH / 100) * 100;

    var svg = el('svg', {
      viewBox: '0 0 ' + W + ' ' + H,
      class: 'chart-svg',
      preserveAspectRatio: 'xMidYMid meet'
    });

    var plotW = W - pad.l - pad.r;
    var plotH = H - pad.t - pad.b;

    function sx(x) { return pad.l + (x / xMax) * plotW; }
    function sy(y) { return pad.t + plotH - (y / yMax) * plotH; }

    var nx = 4, ny = 4;
    for (var i = 0; i <= nx; i++) {
      var gx = pad.l + (i / nx) * plotW;
      svg.appendChild(el('line', {
        x1: gx, y1: pad.t, x2: gx, y2: pad.t + plotH,
        class: 'chart-grid'
      }));
      var tickVal = Math.round((i / nx) * xMax);
      svg.appendChild(txt('text', {
        x: gx, y: pad.t + plotH + 14,
        class: 'chart-tick', 'text-anchor': 'middle'
      }, tickVal));
    }
    for (var j = 0; j <= ny; j++) {
      var gy = pad.t + (j / ny) * plotH;
      svg.appendChild(el('line', {
        x1: pad.l, y1: gy, x2: pad.l + plotW, y2: gy,
        class: 'chart-grid'
      }));
      var tickV = Math.round(((ny - j) / ny) * yMax);
      svg.appendChild(txt('text', {
        x: pad.l - 6, y: gy + 4,
        class: 'chart-tick', 'text-anchor': 'end'
      }, tickV));
    }

    svg.appendChild(txt('text', {
      x: pad.l + plotW / 2, y: H - 4,
      class: 'chart-axis-label', 'text-anchor': 'middle'
    }, 'width (px)'));
    svg.appendChild(txt('text', {
      x: 10, y: pad.t + plotH / 2,
      class: 'chart-axis-label', 'text-anchor': 'middle',
      transform: 'rotate(-90 10 ' + (pad.t + plotH / 2) + ')'
    }, 'height (px)'));

    var rMin = 3, rMax = 18;
    data.forEach(function (d) {
      var r = rMin + (Math.sqrt(d.count / maxC)) * (rMax - rMin);
      var g = el('g', { class: 'chart-dot' });
      g.appendChild(el('circle', {
        cx: sx(d.w), cy: sy(d.h), r: r,
        class: 'chart-dot-circle'
      }));
      var title = el('title');
      title.textContent = d.w + '×' + d.h + ' — ' + d.count + ' videos';
      g.appendChild(title);
      svg.appendChild(g);
    });

    container.appendChild(svg);
  }

  function renderDurationBars(container, payload) {
    container.innerHTML = '';
    var bins = payload && payload.bins ? payload.bins : [];
    if (!bins.length) return;

    var totalBins = 0;
    bins.forEach(function (b) { totalBins += b.count; });
    var minBin = Math.max(1, Math.ceil(totalBins * 0.01));
    var lo = 0, hi = bins.length - 1;
    while (lo < hi && bins[lo].count < minBin) lo++;
    while (hi > lo && bins[hi].count < minBin) hi--;
    bins = bins.slice(lo, hi + 1);

    var unit = payload.unit || 's';
    var W = 360, H = 220;
    var pad = { l: 36, r: 14, t: 12, b: 38 };
    var maxC = 0;
    bins.forEach(function (b) { if (b.count > maxC) maxC = b.count; });
    if (maxC === 0) maxC = 1;

    var svg = el('svg', {
      viewBox: '0 0 ' + W + ' ' + H,
      class: 'chart-svg',
      preserveAspectRatio: 'xMidYMid meet'
    });

    var plotW = W - pad.l - pad.r;
    var plotH = H - pad.t - pad.b;
    var n = bins.length;
    var bw = (plotW / n) * 0.78;
    var bgap = (plotW / n) * 0.22;

    var ny = 4;
    for (var j = 0; j <= ny; j++) {
      var gy = pad.t + (j / ny) * plotH;
      svg.appendChild(el('line', {
        x1: pad.l, y1: gy, x2: pad.l + plotW, y2: gy,
        class: 'chart-grid'
      }));
      var tickV = Math.round(((ny - j) / ny) * maxC);
      svg.appendChild(txt('text', {
        x: pad.l - 6, y: gy + 4,
        class: 'chart-tick', 'text-anchor': 'end'
      }, tickV));
    }

    var total = 0;
    bins.forEach(function (b) { total += b.count; });

    bins.forEach(function (b, i) {
      var x = pad.l + i * (plotW / n) + bgap / 2;
      var hRatio = b.count / maxC;
      var bh = hRatio * plotH;
      var y = pad.t + plotH - bh;
      var g = el('g', { class: 'chart-bar' });
      g.appendChild(el('rect', {
        x: x, y: y, width: bw, height: bh,
        rx: 2, ry: 2,
        class: 'chart-bar-rect'
      }));
      if (b.count > 0) {
        svg.appendChild(txt('text', {
          x: x + bw / 2, y: y - 3,
          class: 'chart-bar-value', 'text-anchor': 'middle'
        }, b.count));
      }
      var label = b.lo + '–' + b.hi;
      svg.appendChild(txt('text', {
        x: x + bw / 2, y: pad.t + plotH + 14,
        class: 'chart-tick', 'text-anchor': 'middle'
      }, label));

      var title = el('title');
      var pct = total > 0 ? ((b.count / total) * 100).toFixed(1) : '0';
      title.textContent = b.lo + '–' + b.hi + unit + ' — ' + b.count + ' videos (' + pct + '%)';
      g.appendChild(title);
      svg.appendChild(g);
    });

    svg.appendChild(txt('text', {
      x: pad.l + plotW / 2, y: H - 4,
      class: 'chart-axis-label', 'text-anchor': 'middle'
    }, 'duration (' + unit + ')'));

    container.appendChild(svg);
  }

  function renderAll() {
    document.querySelectorAll('.bench-charts').forEach(function (wrap) {
      var dataNode = wrap.querySelector('.bench-chart-data');
      if (!dataNode) return;
      var payload;
      try { payload = JSON.parse(dataNode.textContent); }
      catch (e) { return; }

      wrap.querySelectorAll('.bench-chart-svg').forEach(function (slot) {
        var t = slot.getAttribute('data-chart-type');
        if (t === 'resolution') renderResolutionScatter(slot, payload.resolution);
        else if (t === 'duration') renderDurationBars(slot, payload.duration);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderAll);
  } else {
    renderAll();
  }
})();
