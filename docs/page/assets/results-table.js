(function () {
  var CSV_URL = 'assets/results.csv';
  var MODELS = [
    'LLaVA-OneVision-2',
    'Qwen3-VL-8B',
    'Keye-VL-1.5-8B',
    'InternVL-3.5-8B',
    'PLM-8B',
    'LLaVA-OneVision-1.5'
  ];
  var DISPLAY_TO_CSV = {
    'VideoMME (sub)': 'VideoMME (w/ s...)',
    'VideoMME-v2 (sub)': 'VideoMME-v2-64',
    'PixMo-Count': 'Pixmo_Count',
    'MeViS-U (J&F)': 'MeViS_U (J&F)'
  };

  function parseCsv(text) {
    var rows = [];
    var row = [];
    var field = '';
    var inQuotes = false;
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      if (ch === '"') {
        if (inQuotes && text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === ',' && !inQuotes) {
        row.push(field);
        field = '';
      } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(field);
        if (row.some(function (cell) { return cell.trim() !== ''; })) rows.push(row);
        row = [];
        field = '';
      } else {
        field += ch;
      }
    }
    if (field || row.length) {
      row.push(field);
      if (row.some(function (cell) { return cell.trim() !== ''; })) rows.push(row);
    }
    return rows;
  }

  function slugText(text) {
    return text
      .replace(/\s+/g, ' ')
      .replace(/[\u00a0]/g, ' ')
      .trim();
  }

  function rowBenchmarkName(row) {
    var first = row.cells[0];
    if (!first) return '';
    var clone = first.cloneNode(true);
    clone.querySelectorAll('button,svg').forEach(function (node) { node.remove(); });
    var label = slugText(clone.textContent);
    if (!label && row.classList.contains('bench-avg')) label = 'Average';
    return label;
  }

  function formatScore(value) {
    return Number(value).toFixed(1);
  }

  function setScoreCell(cell, value) {
    if (!cell || !isFinite(value)) return;
    cell.textContent = formatScore(value);
  }

  function buildResults(rows) {
    var header = rows[0];
    var modelCols = MODELS.map(function (model) { return header.indexOf(model); });
    var results = { byBenchmark: {}, byCategory: {} };
    rows.slice(1).forEach(function (row) {
      var category = row[0];
      var benchmark = row[1];
      if (!category || !benchmark) return;
      var values = modelCols.map(function (idx) { return Number(row[idx]); });
      if (!results.byCategory[category]) results.byCategory[category] = {};
      results.byCategory[category][benchmark] = values;
      results.byBenchmark[benchmark] = values;
    });
    return results;
  }

  function scoreValuesFromRow(row) {
    return Array.prototype.slice.call(row.cells).slice(1, 7).map(function (cell) {
      return Number(slugText(cell.textContent));
    });
  }

  function updateAverageRows(table) {
    var bodyRows = Array.prototype.slice.call(table.querySelectorAll('tr.bench-row'));
    var scoreRows = bodyRows.filter(function (row) {
      return !row.classList.contains('bench-avg') && row.cells.length >= 7;
    });
    var avgRow = bodyRows.find(function (row) { return row.classList.contains('bench-avg'); });
    if (!scoreRows.length || !avgRow) return;
    var sums = [0, 0, 0, 0, 0, 0];
    var count = 0;
    scoreRows.forEach(function (row) {
      var values = scoreValuesFromRow(row);
      if (values.every(isFinite)) {
        values.forEach(function (value, idx) { sums[idx] += value; });
        count++;
      }
    });
    if (!count) return;
    var avgs = sums.map(function (sum) { return Math.round((sum / count) * 10) / 10; });
    var maxValue = avgs.reduce(function (m, v) { return Math.max(m, v); }, 0);
    Array.prototype.slice.call(avgRow.cells).slice(1, 7).forEach(function (cell, idx) {
      setScoreCell(cell, avgs[idx]);
    });
  }

  function updateRows(results) {
    document.querySelectorAll('.bench-table').forEach(function (table) {
      table.querySelectorAll('tr.bench-row').forEach(function (row) {
        if (row.classList.contains('bench-avg')) return;
        var label = rowBenchmarkName(row);
        var csvKey = DISPLAY_TO_CSV[label] === undefined ? label : DISPLAY_TO_CSV[label];
        var values = results.byBenchmark[csvKey];
        if (values) {
          var maxValue = values.reduce(function (m, v) { return isFinite(v) ? Math.max(m, v) : m; }, 0);
          Array.prototype.slice.call(row.cells).slice(1, 7).forEach(function (cell, idx) {
            setScoreCell(cell, values[idx]);
          });
        } else {
          var staticValues = scoreValuesFromRow(row);
          var staticMax = staticValues.reduce(function (m, v) { return isFinite(v) ? Math.max(m, v) : m; }, 0);
          if (!staticMax) return;
          staticValues.forEach(function (value, idx) {
            setScoreCell(row.cells[idx + 1], value);
          });
        }
      });
      updateAverageRows(table);
    });
  }

  function markLoaded() {
    document.documentElement.classList.add('bench-results-loaded');
  }

  function loadResults() {
    return fetch(CSV_URL, { cache: 'no-store' })
      .then(function (res) {
        if (!res.ok) throw new Error('Failed to load ' + CSV_URL + ': ' + res.status);
        return res.text();
      })
      .then(function (text) {
        var rows = parseCsv(text);
        if (rows.length < 2) throw new Error('No benchmark rows found in ' + CSV_URL);
        updateRows(buildResults(rows));
        markLoaded();
      })
      .catch(function (err) {
        console.warn('[results-table]', err);
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadResults);
  } else {
    loadResults();
  }
})();
