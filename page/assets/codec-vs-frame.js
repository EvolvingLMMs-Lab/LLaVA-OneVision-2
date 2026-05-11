(function () {
  var SVG_NS = 'http://www.w3.org/2000/svg';

  var DATASETS = [
    {
      key: 'qvhighlights',
      name: 'QVHighlights',
      yMin: 10, yMax: 70, yStep: 10,
      frames: [4, 8, 16, 32, 64],
      frame: [12.3, 23.3, 24.9, 53.2, 61.7],
      codec: [27.7, 40.4, 51.5, 59.2, 63.5]
    },
    {
      key: 'charades',
      name: 'Charades-STA',
      yMin: 15, yMax: 55, yStep: 10,
      frames: [4, 8, 16, 32, 64],
      frame: [17.4, 37.1, 48.1, 53.1, 53.5],
      codec: [42.4, 48.2, 51.0, 49.9, 50.1]
    },
    {
      key: 'activitynet',
      name: 'ActivityNet',
      yMin: 10, yMax: 55, yStep: 10,
      frames: [4, 8, 16, 32, 64],
      frame: [12.0, 18.5, 28.1, 39.9, 48.9],
      codec: [23.1, 31.8, 40.6, 47.5, 51.2]
    },
    {
      key: 'lvbench',
      name: 'LVBench',
      yMin: 36, yMax: 52, yStep: 4,
      frames: [16, 32, 64, 128],
      frame: [38.9, 41.6, 45.6, 47.7],
      codec: [40.9, 42.9, 47.1, 49.5]
    },
    {
      key: 'videomme-long-sub',
      name: 'VideoMME-long (w/ sub)',
      yMin: 50, yMax: 70, yStep: 4,
      frames: [8, 16, 32, 64, 128],
      frame: [50.7, 55.0, 57.3, 59.2, 63.2],
      codec: [56.9, 59.2, 63.1, 66.4, 67.6]
    },
    {
      key: 'videoeval-pro',
      name: 'VideoEval-Pro',
      yMin: 42, yMax: 58, yStep: 4,
      frames: [8, 16, 32, 64, 128],
      frame: [43.1, 44.8, 48.6, 52.3, 54.0],
      codec: [46.2, 48.2, 51.4, 53.7, 55.5]
    },
    {
      key: 'jumpscore',
      name: 'JumpScore',
      yMin: 35, yMax: 65, yStep: 5,
      frames: [16, 32, 64, 128],
      frame: [37.4, 36.2, 37.7, 42.8],
      codec: [40.2, 50.3, 59.1, 61.8]
    }
  ];

  var COLOR_CODEC = '#2563eb';
  var COLOR_FRAME = '#94a3b8';
  var COLOR_DELTA = '#0d9488';
  var COLOR_DELTA_NEG = '#dc2626';

  function el(tag, attrs) {
    var n = document.createElementNS(SVG_NS, tag);
    if (attrs) for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }
  function txt(tag, attrs, content) {
    var n = el(tag, attrs);
    n.textContent = content;
    return n;
  }

  function xScale(frame, frames, w) {
    var lo = Math.log2(frames[0]), hi = Math.log2(frames[frames.length - 1]);
    return ((Math.log2(frame) - lo) / (hi - lo)) * w;
  }
  function yScale(v, ds, h) {
    return h - ((v - ds.yMin) / (ds.yMax - ds.yMin)) * h;
  }

  var tip = null;
  function ensureTip(host) {
    if (tip) return tip;
    tip = document.createElement('div');
    tip.className = 'cvf-tooltip';
    tip.style.cssText = [
      'position:absolute',
      'pointer-events:none',
      'background:#0f172a',
      'color:#f8fafc',
      'padding:8px 10px',
      'border-radius:6px',
      'font-size:11px',
      'line-height:1.4',
      'box-shadow:0 4px 12px rgba(15,23,42,0.18)',
      'opacity:0',
      'transition:opacity 0.12s ease',
      'z-index:50',
      'white-space:nowrap',
      'font-family:-apple-system,BlinkMacSystemFont,Inter,Segoe UI,sans-serif'
    ].join(';');
    host.appendChild(tip);
    return tip;
  }
  function showTip(host, x, y, html) {
    var t = ensureTip(host);
    t.innerHTML = html;
    var hostRect = host.getBoundingClientRect();
    var tw = t.offsetWidth, th = t.offsetHeight;
    var px = x - tw / 2;
    var py = y - th - 12;
    if (px < 4) px = 4;
    if (px + tw > hostRect.width - 4) px = hostRect.width - tw - 4;
    if (py < 4) py = y + 14;
    t.style.left = px + 'px';
    t.style.top = py + 'px';
    t.style.opacity = '1';
  }
  function hideTip() {
    if (tip) tip.style.opacity = '0';
  }

  function svgPointToHost(svg, host, sx, sy) {
    var rect = svg.getBoundingClientRect();
    var hostRect = host.getBoundingClientRect();
    var scaleX = rect.width / svg.viewBox.baseVal.width;
    var scaleY = rect.height / svg.viewBox.baseVal.height;
    return {
      x: (rect.left - hostRect.left) + sx * scaleX,
      y: (rect.top - hostRect.top) + sy * scaleY
    };
  }

  function renderPanel(svg, ds, ox, oy, host, state, panelW, opts) {
    opts = opts || {};
    panelW = panelW || 246;
    var leftPad = opts.leftPad || 42;
    var rightPad = opts.rightPad || 22;
    if (panelW <= 250) rightPad = panelW - 170 - leftPad;
    if (rightPad < 18) rightPad = 18;
    var INNER_W = panelW - leftPad - rightPad;
    var INNER_H = opts.innerH || 180;
    var INNER_OX = ox + leftPad;
    var INNER_OY = oy + (opts.innerTop || 60);
    var frames = ds.frames;

    var g = el('g');
    svg.appendChild(g);

    var headerCx = ox + panelW / 2;
    g.appendChild(txt('text', {
      x: headerCx, y: oy + 22,
      'class': 'cvf-panel-title', 'text-anchor': 'middle'
    }, ds.name));

    var lastIdx = frames.length - 1;
    var d4 = ds.codec[0] - ds.frame[0];
    var d64 = ds.codec[lastIdx] - ds.frame[lastIdx];
    function deltaSpan(v) {
      var col = v >= 0 ? COLOR_DELTA : COLOR_DELTA_NEG;
      var s = (v > 0 ? '+' : '') + v.toFixed(1);
      return '<tspan font-weight="700" fill="' + col + '">' + s + '</tspan>';
    }
    var subText = el('text', {
      x: headerCx, y: oy + 40,
      'class': 'cvf-panel-sub', 'text-anchor': 'middle'
    });
    if (opts.compactSub) {
      subText.innerHTML = frames[0] + 'f: ' + deltaSpan(d4) +
        '\u00A0\u00B7\u00A0' + frames[lastIdx] + 'f: ' + deltaSpan(d64);
    } else {
      subText.innerHTML = '@ ' + frames[0] + ' frames: ' + deltaSpan(d4) +
        '\u00A0\u00A0\u00B7\u00A0\u00A0' +
        '@ ' + frames[lastIdx] + ' frames: ' + deltaSpan(d64);
    }
    g.appendChild(subText);

    var inner = el('g', { transform: 'translate(' + INNER_OX + ',' + INNER_OY + ')' });
    g.appendChild(inner);

    var nY = Math.round((ds.yMax - ds.yMin) / ds.yStep);
    for (var i = 0; i <= nY; i++) {
      var yv = ds.yMin + i * ds.yStep;
      var py = INNER_H - (i / nY) * INNER_H;
      inner.appendChild(el('line', {
        x1: 0, y1: py, x2: INNER_W, y2: py, 'class': 'cvf-grid'
      }));
      inner.appendChild(txt('text', {
        x: -6, y: py + 3, 'class': 'cvf-axis-label', 'text-anchor': 'end'
      }, String(yv)));
    }

    frames.forEach(function (f) {
      var px = xScale(f, frames, INNER_W);
      inner.appendChild(el('line', {
        x1: px, y1: 0, x2: px, y2: INNER_H, 'class': 'cvf-grid'
      }));
      inner.appendChild(txt('text', {
        x: px, y: INNER_H + 14, 'class': 'cvf-axis-label', 'text-anchor': 'middle'
      }, String(f)));
    });

    inner.appendChild(el('line', { x1: 0, y1: 0, x2: 0, y2: INNER_H, 'class': 'cvf-axis' }));
    inner.appendChild(el('line', { x1: 0, y1: INNER_H, x2: INNER_W, y2: INNER_H, 'class': 'cvf-axis' }));

    if (!opts.hideAxisTitles) {
      inner.appendChild(txt('text', {
        x: INNER_W / 2, y: INNER_H + 32, 'class': 'cvf-axis-title', 'text-anchor': 'middle'
      }, 'frame budget (log scale)'));
      inner.appendChild(txt('text', {
        transform: 'translate(-36,' + (INNER_H / 2) + ') rotate(-90)',
        'class': 'cvf-axis-title', 'text-anchor': 'middle'
      }, 'metric'));
    }

    var codecPts = frames.map(function (f, i) {
      return { x: xScale(f, frames, INNER_W), y: yScale(ds.codec[i], ds, INNER_H), v: ds.codec[i], f: f };
    });
    var framePts = frames.map(function (f, i) {
      return { x: xScale(f, frames, INNER_W), y: yScale(ds.frame[i], ds, INNER_H), v: ds.frame[i], f: f };
    });
    var codecStr = codecPts.map(function (p) { return p.x + ',' + p.y; }).join(' ');
    var frameStr = framePts.map(function (p) { return p.x + ',' + p.y; }).join(' ');

    var poly = framePts.map(function (p) { return p.x + ',' + p.y; })
      .concat(codecPts.slice().reverse().map(function (p) { return p.x + ',' + p.y; }))
      .join(' ');
    inner.appendChild(el('polygon', {
      points: poly, fill: COLOR_CODEC, opacity: 0.08, 'class': 'cvf-area'
    }));

    inner.appendChild(el('polyline', {
      points: frameStr, fill: 'none', stroke: COLOR_FRAME,
      'stroke-width': 2, 'stroke-dasharray': '4 3',
      'class': 'cvf-line cvf-line-frame cvf-series-frame'
    }));
    inner.appendChild(el('polyline', {
      points: codecStr, fill: 'none', stroke: COLOR_CODEC,
      'stroke-width': 2.2,
      'class': 'cvf-line cvf-line-codec cvf-series-codec'
    }));

    var pinLine = el('line', {
      x1: 0, y1: 0, x2: 0, y2: INNER_H,
      stroke: '#0f172a', 'stroke-width': 1, 'stroke-dasharray': '3 3',
      opacity: 0,
      'class': 'cvf-pin-line'
    });
    inner.appendChild(pinLine);

    var hoverLine = el('line', {
      x1: 0, y1: 0, x2: 0, y2: INNER_H,
      stroke: '#94a3b8', 'stroke-width': 1,
      opacity: 0,
      'class': 'cvf-hover-line',
      'pointer-events': 'none'
    });
    inner.appendChild(hoverLine);

    function makePoint(p, kind) {
      var isCodec = kind === 'codec';
      var c = el('circle', {
        cx: p.x, cy: p.y,
        r: isCodec ? 4 : 3.5,
        fill: isCodec ? COLOR_CODEC : '#ffffff',
        stroke: isCodec ? COLOR_CODEC : COLOR_FRAME,
        'stroke-width': isCodec ? 0 : 1.6,
        'class': 'cvf-pt cvf-pt-' + kind + ' cvf-series-' + kind,
        'data-frame': p.f,
        'data-kind': kind,
        style: 'cursor:pointer;transition:r 0.15s ease,stroke-width 0.15s ease;'
      });
      var hit = el('circle', {
        cx: p.x, cy: p.y, r: 14, fill: 'transparent',
        'class': 'cvf-hit cvf-series-' + kind,
        style: 'cursor:pointer;'
      });

      function emphasize() {
        c.setAttribute('r', isCodec ? 6 : 5.5);
        c.setAttribute('stroke-width', isCodec ? 2 : 2.4);
        if (!isCodec) c.setAttribute('stroke', '#475569');
        hoverLine.setAttribute('x1', p.x);
        hoverLine.setAttribute('x2', p.x);
        hoverLine.setAttribute('opacity', 0.6);
      }
      function relax() {
        c.setAttribute('r', isCodec ? 4 : 3.5);
        c.setAttribute('stroke-width', isCodec ? 0 : 1.6);
        c.setAttribute('stroke', isCodec ? COLOR_CODEC : COLOR_FRAME);
        hoverLine.setAttribute('opacity', 0);
      }

      function showAt() {
        emphasize();
        var idx = frames.indexOf(p.f);
        var codecV = ds.codec[idx];
        var frameV = ds.frame[idx];
        var delta = codecV - frameV;
        var deltaCol = delta >= 0 ? COLOR_DELTA : COLOR_DELTA_NEG;
        var deltaStr = (delta > 0 ? '+' : '') + delta.toFixed(1);
        var html =
          '<div style="font-weight:700;font-size:11px;margin-bottom:4px;color:#f8fafc;">' +
            ds.name + ' &middot; ' + p.f + ' frames' +
          '</div>' +
          '<div style="display:grid;grid-template-columns:auto auto;gap:2px 12px;font-size:10px;">' +
            '<span style="color:' + COLOR_CODEC + ';">&#9679; codec</span>' +
            '<span style="font-weight:700;text-align:right;">' + codecV.toFixed(1) + '</span>' +
            '<span style="color:#cbd5e1;">&#9675; uniform</span>' +
            '<span style="font-weight:700;text-align:right;">' + frameV.toFixed(1) + '</span>' +
            '<span style="color:' + deltaCol + ';">&Delta;</span>' +
            '<span style="font-weight:700;text-align:right;color:' + deltaCol + ';">' + deltaStr + '</span>' +
          '</div>';
        var sx = INNER_OX + p.x;
        var sy = INNER_OY + p.y;
        var hp = svgPointToHost(svg, host, sx, sy);
        showTip(host, hp.x, hp.y, html);
      }

      hit.addEventListener('mouseenter', showAt);
      hit.addEventListener('mouseleave', function () { relax(); hideTip(); });
      hit.addEventListener('click', function (evt) {
        evt.stopPropagation();
        pinLine.setAttribute('x1', p.x);
        pinLine.setAttribute('x2', p.x);
        pinLine.setAttribute('opacity', 0.5);
        state.pinned = { ds: ds.key, frame: p.f };
        showAt();
      });

      inner.appendChild(c);
      inner.appendChild(hit);
    }

    framePts.forEach(function (p) { makePoint(p, 'frame'); });
    codecPts.forEach(function (p) { makePoint(p, 'codec'); });

    function addLabel(p, color, side) {
      var anchor = side === 'left' ? 'start' : 'end';
      var dx = side === 'left' ? 6 : -6;
      var dy = -6;
      inner.appendChild(txt('text', {
        x: p.x + dx, y: p.y + dy,
        'class': 'cvf-data-label', fill: color, 'text-anchor': anchor
      }, p.v.toFixed(1)));
    }
    if (!opts.hideDataLabels) {
      addLabel(codecPts[0], COLOR_CODEC, 'left');
      addLabel(framePts[0], '#64748b', 'left');
      addLabel(codecPts[lastIdx], COLOR_CODEC, 'right');
      addLabel(framePts[lastIdx], '#64748b', 'right');
    }

    state.unpinFns = state.unpinFns || [];
    state.unpinFns.push(function () {
      pinLine.setAttribute('opacity', 0);
    });
  }

  function getLayout(vw) {
    var n = DATASETS.length;
    var cols, panelStep, panelW, headerH, legendArea, sidePad, rowH, compactHeader;
    if (vw <= 480) {
      cols = 1;
      panelW = vw - 16;
      if (panelW < 280) panelW = 280;
      if (panelW > 360) panelW = 360;
      panelStep = panelW;
      sidePad = 8;
      headerH = 64;
      legendArea = 56;
      rowH = 240;
      compactHeader = true;
    } else if (vw <= 600) {
      cols = 1;
      panelW = 360;
      panelStep = panelW;
      sidePad = 16;
      headerH = 70;
      legendArea = 56;
      rowH = 270;
      compactHeader = true;
    } else if (vw <= 900) {
      cols = 2;
      panelW = 340;
      panelStep = panelW;
      sidePad = 24;
      headerH = 86;
      legendArea = 56;
      rowH = 280;
      compactHeader = true;
    } else {
      cols = 4;
      panelW = 246;
      panelStep = panelW;
      sidePad = 60;
      headerH = 110;
      legendArea = 32;
      rowH = 326;
      compactHeader = false;
    }
    var rows = Math.ceil(n / cols);
    var W = sidePad * 2 + cols * panelStep;
    var H = headerH + rows * rowH + legendArea;
    return {
      cols: cols, rows: rows,
      panelStep: panelStep, panelW: panelW,
      sidePad: sidePad, headerH: headerH, rowH: rowH,
      W: W, H: H, compactHeader: compactHeader
    };
  }

  function renderCarousel(host) {
    host.innerHTML = '';
    host.style.position = 'relative';
    host.style.width = '100%';
    host.style.maxWidth = '100%';
    host.style.minWidth = '0';
    host.style.overflow = 'hidden';
    host.style.boxSizing = 'border-box';

    var SVG_W = 260;
    var SVG_H = 244;

    var scroller = document.createElement('div');
    scroller.style.cssText = [
      'display:flex',
      'width:100%',
      'max-width:100%',
      'min-width:0',
      'overflow-x:auto',
      'overflow-y:hidden',
      'scroll-snap-type:x mandatory',
      '-webkit-overflow-scrolling:touch',
      'gap:0',
      'padding:0',
      'margin:0',
      'box-sizing:border-box',
      'scrollbar-width:none'
    ].join(';');
    var styleEl = document.createElement('style');
    styleEl.textContent = '.cvf-mobile-scroller::-webkit-scrollbar{display:none}';
    if (!document.getElementById('cvf-mobile-style')) {
      styleEl.id = 'cvf-mobile-style';
      document.head.appendChild(styleEl);
    }
    scroller.className = 'cvf-mobile-scroller';
    host.appendChild(scroller);

    var state = { pinned: null, unpinFns: [] };
    var slides = [];

    DATASETS.forEach(function (ds) {
      var slide = document.createElement('div');
      slide.style.cssText = [
        'flex:0 0 100%',
        'min-width:0',
        'width:100%',
        'scroll-snap-align:center',
        'background:#ffffff',
        'border:1px solid #e2e8f0',
        'border-radius:8px',
        'padding:4px 2px',
        'box-sizing:border-box',
        'overflow:hidden'
      ].join(';');

      var svgWrap = document.createElement('div');
      svgWrap.style.cssText = 'width:100%;overflow:hidden;';

      var svg = el('svg', {
        xmlns: SVG_NS,
        viewBox: '0 0 ' + SVG_W + ' ' + SVG_H,
        preserveAspectRatio: 'xMidYMid meet',
        width: '100%',
        height: 'auto',
        'font-family': "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif",
        style: 'width:100%;height:auto;display:block;max-width:100%;overflow:hidden;'
      });
      svg.appendChild(el('rect', { width: SVG_W, height: SVG_H, fill: '#ffffff' }));

      renderPanel(svg, ds, 0, 0, host, state, SVG_W, {
        leftPad: 34,
        rightPad: 18,
        innerTop: 54,
        innerH: 168,
        compactSub: true,
        hideAxisTitles: true,
        hideDataLabels: true
      });

      svgWrap.appendChild(svg);
      slide.appendChild(svgWrap);
      scroller.appendChild(slide);
      slides.push(slide);
    });

    var dotsWrap = document.createElement('div');
    dotsWrap.style.cssText = 'display:flex;justify-content:center;gap:5px;padding:5px 0 0;';
    var dots = DATASETS.map(function (_, i) {
      var d = document.createElement('button');
      d.type = 'button';
      d.setAttribute('aria-label', 'Show panel ' + (i + 1));
      d.style.cssText = [
        'width:6px', 'height:6px', 'padding:0', 'border:none',
        'border-radius:50%', 'background:#cbd5e1',
        'cursor:pointer', 'transition:background .15s ease, transform .15s ease'
      ].join(';');
      d.addEventListener('click', function () {
        slides[i].scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      });
      dotsWrap.appendChild(d);
      return d;
    });
    host.appendChild(dotsWrap);

    function updateDots() {
      var center = scroller.scrollLeft + scroller.clientWidth / 2;
      var bestI = 0, bestD = Infinity;
      slides.forEach(function (s, i) {
        var c = s.offsetLeft + s.offsetWidth / 2;
        var d = Math.abs(c - center);
        if (d < bestD) { bestD = d; bestI = i; }
      });
      dots.forEach(function (d, i) {
        if (i === bestI) {
          d.style.background = '#2563eb';
          d.style.transform = 'scale(1.4)';
        } else {
          d.style.background = '#cbd5e1';
          d.style.transform = 'scale(1)';
        }
      });
    }
    scroller.addEventListener('scroll', function () {
      requestAnimationFrame(updateDots);
    });
    setTimeout(updateDots, 50);

    host.addEventListener('click', function (evt) {
      var tag = evt.target && evt.target.tagName;
      if (tag === 'svg' || tag === 'rect' || tag === 'DIV') {
        state.unpinFns.forEach(function (fn) { fn(); });
        state.pinned = null;
        hideTip();
      }
    });
  }

  function render(host) {
    var vw = host.clientWidth || window.innerWidth || 1080;
    if (vw <= 600) {
      renderCarousel(host);
      return;
    }

    host.innerHTML = '';
    host.style.position = 'relative';

    var L = getLayout(vw);
    var W = L.W, H = L.H;

    var svg = el('svg', {
      xmlns: SVG_NS,
      viewBox: '0 0 ' + W + ' ' + H,
      'font-family': "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif",
      style: 'width:100%;height:auto;display:block;'
    });
    svg.appendChild(el('rect', { width: W, height: H, fill: '#ffffff' }));

    if (L.compactHeader) {
      svg.appendChild(txt('text', {
        x: L.sidePad, y: 22, 'class': 'cvf-subtitle'
      }, 'Figure 3 \u00B7 Codec vs Frame'));
      svg.appendChild(txt('text', {
        x: L.sidePad, y: 44, 'class': 'cvf-title',
        style: 'font-size:14px;'
      }, 'Codec sampling unlocks low-frame regimes.'));
      svg.appendChild(el('line', {
        x1: L.sidePad, y1: L.headerH - 8, x2: W - L.sidePad, y2: L.headerH - 8,
        'class': 'cvf-divider'
      }));
    } else {
      svg.appendChild(txt('text', {
        x: L.sidePad, y: 32, 'class': 'cvf-subtitle'
      }, 'Figure 3 \u00B7 Codec vs Frame Sampling'));
      svg.appendChild(txt('text', {
        x: L.sidePad, y: 58, 'class': 'cvf-title'
      }, 'Same token budget. Codec sampling unlocks low-frame regimes.'));
      var subText = el('text', { x: L.sidePad, y: 80, 'class': 'cvf-panel-sub' });
      subText.innerHTML = 'Across four temporal grounding benchmarks, codec-aligned sampling matches or exceeds uniform frame sampling, with the largest gains at <tspan font-weight="700" fill="#0d9488">low frame budgets</tspan> where uniform sampling starves the model. <tspan fill="#94a3b8">Hover any point. Click legend to toggle a series.</tspan>';
      svg.appendChild(subText);
      svg.appendChild(el('line', {
        x1: L.sidePad, y1: 100, x2: W - L.sidePad, y2: 100, 'class': 'cvf-divider'
      }));
    }

    var state = { pinned: null, unpinFns: [] };

    DATASETS.forEach(function (ds, i) {
      var col = i % L.cols;
      var row = Math.floor(i / L.cols);
      var ox = L.sidePad + col * L.panelStep;
      var oy = L.headerH + row * L.rowH;
      renderPanel(svg, ds, ox, oy, host, state, L.panelW);
    });

    var legY = L.headerH + L.rows * L.rowH + (L.cols === 4 ? 26 : 36);

    function makeLegend(x, kind, label) {
      var g = el('g', {
        'class': 'cvf-legend-item cvf-legend-' + kind,
        style: 'cursor:pointer;'
      });
      if (kind === 'codec') {
        g.appendChild(el('line', {
          x1: x, y1: legY, x2: x + 22, y2: legY,
          stroke: COLOR_CODEC, 'stroke-width': 2.2
        }));
        g.appendChild(el('circle', { cx: x + 11, cy: legY, r: 3.5, fill: COLOR_CODEC }));
      } else {
        g.appendChild(el('line', {
          x1: x, y1: legY, x2: x + 22, y2: legY,
          stroke: COLOR_FRAME, 'stroke-width': 2, 'stroke-dasharray': '4 3'
        }));
        g.appendChild(el('circle', {
          cx: x + 11, cy: legY, r: 3,
          fill: '#ffffff', stroke: COLOR_FRAME, 'stroke-width': 1.6
        }));
      }
      g.appendChild(txt('text', { x: x + 30, y: legY + 4, 'class': 'cvf-legend' }, label));
      g.addEventListener('click', function () {
        var hidden = svg.classList.toggle('cvf-hide-' + kind);
        g.setAttribute('opacity', hidden ? 0.35 : 1);
      });
      svg.appendChild(g);
    }

    if (L.cols >= 4) {
      makeLegend(235, 'codec', 'Codec-aligned sampling');
      makeLegend(455, 'frame', 'Uniform frame sampling');
      svg.appendChild(el('rect', {
        x: 675, y: legY - 7, width: 22, height: 14, rx: 2,
        fill: COLOR_CODEC, opacity: 0.18
      }));
      svg.appendChild(txt('text', { x: 705, y: legY + 4, 'class': 'cvf-legend' }, 'Codec advantage region'));
    } else {
      var lx = L.sidePad;
      makeLegend(lx, 'codec', 'Codec');
      makeLegend(lx + 84, 'frame', 'Uniform');
      var advX = lx + 178;
      if (advX + 90 > W - L.sidePad) {
        advX = W - L.sidePad - 90;
      }
      svg.appendChild(el('rect', {
        x: advX, y: legY - 7, width: 16, height: 12, rx: 2,
        fill: COLOR_CODEC, opacity: 0.18
      }));
      svg.appendChild(txt('text', {
        x: advX + 22, y: legY + 4, 'class': 'cvf-legend'
      }, 'Advantage'));
    }

    host.appendChild(svg);

    host.addEventListener('click', function (evt) {
      var tag = evt.target && evt.target.tagName;
      if (tag === 'svg' || (tag === 'rect' && Number(evt.target.getAttribute('width')) === W)) {
        state.unpinFns.forEach(function (fn) { fn(); });
        state.pinned = null;
        hideTip();
      }
    });
  }

  var resizeTimer = null;
  var lastCols = null;
  function onResize() {
    var host = document.getElementById('codec-vs-frame-chart');
    if (!host) return;
    var vw = host.clientWidth || window.innerWidth;
    var nextCols = getLayout(vw).cols;
    if (nextCols === lastCols) return;
    lastCols = nextCols;
    render(host);
  }

  function init() {
    var host = document.getElementById('codec-vs-frame-chart');
    if (!host) return;
    lastCols = getLayout(host.clientWidth || window.innerWidth).cols;
    render(host);
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(onResize, 180);
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.CodecVsFrame = { render: render, data: DATASETS };
})();
