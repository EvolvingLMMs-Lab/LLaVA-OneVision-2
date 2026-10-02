(function () {
  'use strict';
  var root = document.getElementById('token-budget-demo');
  if (!root) return;
  // Illustrative, not measured: each GOP spends 9 + 2 + 2 + 1 + 2 + 2 = 18.
  // Three GOPs cover 18 frames with exactly 54 visible token squares.
  var pPatterns = [[0, 3], [3, 4], [4], [4, 7], [7, 8]];
  var full = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  var totalTokens = 54;
  // Keep the original uniform-lane pace: nine patches every 700 ms.
  // Both lanes spend one token on each shared tick, irrespective of frame size.
  var millisecondsPerToken = 700 / 9;
  var lanes = {};
  ['uniform', 'codec'].forEach(function (kind) {
    var host = root.querySelector('[data-frames="' + kind + '"]');
    var frames = [], tokenStart = 0;
    for (var i = 0; i < 18; i++) {
      var isI = i % 6 === 0;
      var selected = kind === 'uniform' ? (i < 6 ? full : []) : (isI ? full : pPatterns[i % 6 - 1]);
      var frame = document.createElement('div');
      frame.className = 'budget-frame' + (kind === 'codec' ? (isI ? ' is-i' : ' is-p') : (i >= 6 ? ' is-unavailable' : ''));
      var patches = [];
      for (var j = 0; j < 9; j++) {
        var patch = document.createElement('i'); patch.className = 'budget-patch';
        frame.appendChild(patch); patches.push(patch);
      }
      var label = document.createElement('span'); label.className = 'budget-frame-label';
      label.textContent = kind === 'codec' ? (isI ? 'I' : 'P') : (i < 6 ? String(i + 1) : '');
      frame.appendChild(label); host.appendChild(frame);
      frames.push({element:frame, patches:patches, selected:selected, tokenStart:tokenStart});
      tokenStart += selected.length;
    }
    lanes[kind] = {frames:frames, tokens:root.querySelector('[data-tokens="' + kind + '"]'),
      meter:root.querySelector('[data-meter="' + kind + '"]'), count:root.querySelector('[data-count="' + kind + '"]'),
      coverage:root.querySelector('[data-coverage="' + kind + '"]')};
  });
  var message = root.querySelector('[data-budget-message]');
  var reduced = matchMedia('(prefers-reduced-motion: reduce)');
  var position = reduced.matches ? totalTokens : 0;
  var running = false, raf = 0, previous = null, lastStage = -1;
  var inView = false, pageActive = true, endHold = 0;
  function bilingual(en, zh) {
    return '<span class="i18n" data-lang="en">' + en + '</span><span class="i18n" data-lang="zh">' + zh + '</span>';
  }
  function render() {
    root.dataset.running = String(running);
    var spent = Math.min(totalTokens, Math.floor(position + 1e-7));
    Object.keys(lanes).forEach(function (kind) {
      var lane = lanes[kind], total = 0, complete = 0;
      lane.frames.forEach(function (frame) {
        var kept = Math.min(frame.selected.length, Math.max(0, spent - frame.tokenStart));
        total += kept;
        if (kept && kept === frame.selected.length) complete++;
        frame.element.classList.toggle('is-current', running && position >= frame.tokenStart && position < frame.tokenStart + frame.selected.length);
        frame.patches.forEach(function (patch, j) {
          var rank = frame.selected.indexOf(j);
          patch.classList.toggle('is-kept', rank >= 0 && rank < kept);
        });
      });
      lane.tokens.textContent = total; lane.count.textContent = complete;
      lane.meter.style.transform = 'scaleX(' + total / totalTokens + ')';
      lane.coverage.style.transform = 'scaleX(' + complete / 18 + ')';
    });
    var stage = position >= totalTokens ? 2 : (position >= 9 ? 1 : 0);
    if (stage !== lastStage) {
      lastStage = stage;
      var copy = [
        ['Both rows reveal tokens at the same rate. Nine tokens fill the first frame.', '两行以相同速度显示 token，先用 9 个 token 填满首帧。'],
        ['Same token count. Fewer patches per P-frame let codec selection cover more frames.', '消耗的 token 始终相同；Codec 的 P 帧保留更少 patch，因此推进得更快。'],
        ['54 tokens each. Six full frames, or eighteen codec-aligned frames.', '同样 54 个 token：6 个完整帧，或 18 个 Codec 对齐帧。']
      ][stage];
      message.innerHTML = bilingual(copy[0], copy[1]);
      root.querySelector('[data-step]').textContent = '0' + (stage + 1) + ' / 03';
      root.querySelector('[data-result]').hidden = stage !== 2;
    }
  }
  function pause() {
    running = false; cancelAnimationFrame(raf); raf = 0; previous = null;
    render();
  }
  function tick(now) {
    if (!running) return;
    var delta = previous === null ? 0 : Math.min(Math.max(now - previous, 0), 100);
    previous = now;
    if (position < totalTokens) {
      position = Math.min(totalTokens, position + delta / millisecondsPerToken);
    } else {
      // Leave the completed comparison visible before the next loop.
      endHold += delta;
      if (endHold >= 3500) { position = 0; endHold = 0; }
    }
    render();
    raf = requestAnimationFrame(tick);
  }
  function start() {
    cancelAnimationFrame(raf); running = true; previous = null;
    render(); raf = requestAnimationFrame(tick);
  }
  function syncPlayback() {
    if (!reduced.matches && inView && !document.hidden && pageActive) start();
    else pause();
  }
  document.addEventListener('visibilitychange', syncPlayback);
  window.addEventListener('pagehide', function () { pageActive = false; pause(); });
  window.addEventListener('pageshow', function () { pageActive = true; syncPlayback(); });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      inView = entries[0].isIntersecting; syncPlayback();
    }, {threshold: 0}).observe(root);
  } else { inView = true; }
  reduced.addEventListener('change', function () {
    if (reduced.matches) { position = totalTokens; endHold = 0; }
    syncPlayback();
  });
  // Continuous playback should not repeatedly announce progress to screen readers.
  root.querySelector('.budget-takeaway').setAttribute('aria-live', 'off');
  render(); syncPlayback();
})();
