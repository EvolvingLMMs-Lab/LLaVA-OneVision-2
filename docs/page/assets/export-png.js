(function () {
  var SVG_NS = 'http://www.w3.org/2000/svg';
  var SCALE = 2;
  var H2C_URL = 'https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js';
  var ICON_DOWNLOAD = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';
  var ICON_SPIN = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true" class="exp-spin"><path d="M21 12a9 9 0 1 1-6.22-8.55"/></svg>';

  var h2cPromise = null;
  function loadHtml2Canvas() {
    if (window.html2canvas) return Promise.resolve(window.html2canvas);
    if (h2cPromise) return h2cPromise;
    h2cPromise = new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = H2C_URL;
      s.onload = function () { resolve(window.html2canvas); };
      s.onerror = function () { h2cPromise = null; reject(new Error('html2canvas load failed')); };
      document.head.appendChild(s);
    });
    return h2cPromise;
  }

  function downloadBlob(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function inlineComputedStyles(srcRoot, cloneRoot) {
    var srcAll = srcRoot.querySelectorAll('*');
    var cloneAll = cloneRoot.querySelectorAll('*');
    var STYLE_PROPS = [
      'fill', 'fill-opacity', 'stroke', 'stroke-opacity', 'stroke-width',
      'stroke-dasharray', 'stroke-linecap', 'stroke-linejoin',
      'opacity', 'font-family', 'font-size', 'font-weight',
      'text-anchor', 'letter-spacing', 'visibility', 'display'
    ];
    function applyOne(srcEl, cloneEl) {
      var cs = window.getComputedStyle(srcEl);
      var styleStr = '';
      for (var i = 0; i < STYLE_PROPS.length; i++) {
        var k = STYLE_PROPS[i];
        var v = cs.getPropertyValue(k);
        if (v) styleStr += k + ':' + v + ';';
      }
      cloneEl.setAttribute('style', styleStr);
    }
    applyOne(srcRoot, cloneRoot);
    for (var i = 0; i < srcAll.length; i++) {
      applyOne(srcAll[i], cloneAll[i]);
    }
  }

  function svgToPngBlob(svg) {
    return new Promise(function (resolve, reject) {
      var clone = svg.cloneNode(true);
      inlineComputedStyles(svg, clone);

      clone.querySelectorAll('[style*="stroke-dashoffset"]').forEach(function (n) {
        var s = (n.getAttribute('style') || '')
          .replace(/stroke-dashoffset:[^;]+;?/g, 'stroke-dashoffset:0;');
        n.setAttribute('style', s);
      });

      var vb = svg.viewBox && svg.viewBox.baseVal;
      var w, h;
      if (vb && vb.width && vb.height) {
        w = vb.width;
        h = vb.height;
      } else {
        var r = svg.getBoundingClientRect();
        w = r.width;
        h = r.height;
      }
      clone.setAttribute('xmlns', SVG_NS);
      clone.setAttribute('width', w);
      clone.setAttribute('height', h);
      if (!clone.querySelector('rect[data-bg]')) {
        var bg = document.createElementNS(SVG_NS, 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', '#ffffff');
        bg.setAttribute('data-bg', '1');
        clone.insertBefore(bg, clone.firstChild);
      }

      var src = new XMLSerializer().serializeToString(clone);
      if (!/^<\?xml/.test(src)) src = '<?xml version="1.0" standalone="no"?>\n' + src;
      var blob = new Blob([src], { type: 'image/svg+xml;charset=utf-8' });
      var url = URL.createObjectURL(blob);

      var img = new Image();
      img.onload = function () {
        var canvas = document.createElement('canvas');
        canvas.width = Math.round(w * SCALE);
        canvas.height = Math.round(h * SCALE);
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        canvas.toBlob(function (b) {
          if (b) resolve(b); else reject(new Error('canvas.toBlob failed'));
        }, 'image/png');
      };
      img.onerror = function (e) {
        URL.revokeObjectURL(url);
        reject(new Error('SVG image load failed'));
      };
      img.src = url;
    });
  }

  function elementToPngBlob(el) {
    return loadHtml2Canvas().then(function (h2c) {
      return h2c(el, {
        backgroundColor: '#ffffff',
        scale: SCALE,
        useCORS: true,
        logging: false
      });
    }).then(function (canvas) {
      return new Promise(function (resolve, reject) {
        canvas.toBlob(function (b) {
          if (b) resolve(b); else reject(new Error('canvas.toBlob failed'));
        }, 'image/png');
      });
    });
  }

  function pickSvgInside(container) {
    var svgs = container.querySelectorAll('svg');
    if (svgs.length === 0) return null;
    if (svgs.length === 1) return svgs[0];
    var best = svgs[0], bestArea = 0;
    for (var i = 0; i < svgs.length; i++) {
      var r = svgs[i].getBoundingClientRect();
      var a = r.width * r.height;
      if (a > bestArea) { bestArea = a; best = svgs[i]; }
    }
    return best;
  }

  function makeButton() {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'export-png-btn';
    btn.setAttribute('aria-label', 'Export as PNG');
    btn.title = 'Export as PNG';
    btn.innerHTML = ICON_DOWNLOAD + '<span class="export-png-label">PNG</span>';
    return btn;
  }

  function setBusy(btn, busy) {
    if (busy) {
      btn._origHTML = btn.innerHTML;
      btn.innerHTML = ICON_SPIN + '<span class="export-png-label">…</span>';
      btn.disabled = true;
    } else {
      if (btn._origHTML) btn.innerHTML = btn._origHTML;
      btn.disabled = false;
    }
  }

  function flashError(btn) {
    btn.classList.add('export-png-err');
    setTimeout(function () { btn.classList.remove('export-png-err'); }, 1600);
  }

  function attachExport(container, opts) {
    if (!container || container.querySelector(':scope > .export-png-btn')) return;
    opts = opts || {};
    var btn = makeButton();
    container.style.position = container.style.position || 'relative';
    container.appendChild(btn);

    btn.addEventListener('click', function (evt) {
      evt.stopPropagation();
      setBusy(btn, true);
      var filename = (opts.filename || 'llava-onevision-2-export') + '.png';
      var task;
      if (opts.kind === 'svg') {
        var svg = typeof opts.target === 'function' ? opts.target() : pickSvgInside(container);
        if (!svg) { setBusy(btn, false); flashError(btn); return; }
        task = svgToPngBlob(svg);
      } else {
        var target = typeof opts.target === 'function' ? opts.target() : (opts.target || container);
        task = elementToPngBlob(target);
      }
      task.then(function (blob) {
        downloadBlob(blob, filename);
        setBusy(btn, false);
      }).catch(function (err) {
        console.error('[export-png]', err);
        setBusy(btn, false);
        flashError(btn);
      });
    });
  }

  function slug(s) {
    return String(s || '').trim().toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '') || 'export';
  }

  function captionText(container) {
    var fc = container.querySelector('figcaption');
    if (fc) {
      var en = fc.querySelector('.i18n[data-lang="en"]') || fc;
      return (en.textContent || '').slice(0, 60);
    }
    return '';
  }

  function init() {
    document.querySelectorAll('.method-figure-svg').forEach(function (host, i) {
      var fig = host.closest('figure') || host.parentElement;
      var name = captionText(fig) || ('figure-' + (i + 1));
      attachExport(host, { kind: 'svg', filename: 'llava-onevision-2-' + slug(name) });
    });

    document.querySelectorAll('.bench-card').forEach(function (card) {
      var label = card.querySelector('.bench-caption-label .i18n[data-lang="en"]')
                || card.querySelector('.bench-caption-label')
                || card.querySelector('.bench-caption');
      var name = label ? label.textContent : 'benchmark';
      attachExport(card, {
        kind: 'element',
        target: card,
        filename: 'llava-onevision-2-' + slug(name)
      });
    });

    document.querySelectorAll('.bench-charts').forEach(function (wrap, i) {
      wrap.querySelectorAll('.bench-chart-svg').forEach(function (slot, j) {
        var name = slot.getAttribute('data-chart-title')
                || slot.getAttribute('data-bench')
                || ('chart-' + (i + 1) + '-' + (j + 1));
        attachExport(slot, { kind: 'svg', filename: 'llava-onevision-2-chart-' + slug(name) });
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.ExportPNG = {
    attach: attachExport,
    svgToPngBlob: svgToPngBlob,
    elementToPngBlob: elementToPngBlob,
    download: downloadBlob,
    rescan: init
  };
})();
