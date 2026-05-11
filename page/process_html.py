import re

with open('/ov2/xiangan/sparrow/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

def slugify(text):
    text = text.lower()
    # specifically V*
    text = text.replace('v*', 'v-star')
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def replace_table_body(match):
    tbody_content = match.group(1)
    # Find all tr
    tr_pattern = re.compile(r'(<tr[^>]*>)(.*?)(</tr>)', re.DOTALL)
    
    new_tbody = ""
    for tr_match in tr_pattern.finditer(tbody_content):
        tr_open = tr_match.group(1)
        tr_inner = tr_match.group(2)
        tr_close = tr_match.group(3)
        
        # Extract benchmark name from first td
        td_match = re.search(r'<td>(.*?)</td>', tr_inner)
        if not td_match:
            new_tbody += tr_open + tr_inner + tr_close + "\n"
            continue
            
        bench_name_raw = td_match.group(1)
        # Handle sub tags like MMBench<sub>en</sub>
        bench_name = re.sub(r'<[^>]+>', '', bench_name_raw)
        slug = slugify(bench_name)
        
        # Replace tr_open to add class and data-bench
        # if there are already classes (like bench-section-top), we need to append
        if 'class="' in tr_open:
            tr_open = tr_open.replace('class="', 'class="bench-row ')
        else:
            tr_open = tr_open.replace('<tr', '<tr class="bench-row"')
        
        tr_open = tr_open.replace('>', f' data-bench="{slug}">')
        
        # Add expand button to first td
        button_html = f'<button class="bench-expand" aria-expanded="false" aria-label="Expand"><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5l3 3 3-3"/></svg></button>{bench_name_raw}'
        tr_inner = tr_inner.replace(f'<td>{bench_name_raw}</td>', f'<td>{button_html}</td>', 1)
        
        new_row = tr_open + tr_inner + tr_close
        
        detail_row = f'''
<tr class="bench-detail" data-bench="{slug}">
  <td colspan="7">
    <div class="bench-detail-content">
      <div>
        <p class="i18n" data-lang="en">TBD — description for {bench_name_raw}.</p>
        <p class="i18n" data-lang="zh">TBD — {bench_name_raw} 测试集介绍。</p>
      </div>
    </div>
  </td>
</tr>'''
        new_tbody += "  " + new_row + detail_row + "\n"
        
    return "<tbody>\n" + new_tbody + "                </tbody>"

# We only process the first 3 bench-tables.
# Let's split by '<table class="bench-table">'
parts = html.split('<table class="bench-table">')
if len(parts) >= 4:
    for i in range(1, 4):
        # find tbody
        tbody_match = re.search(r'<tbody>(.*?)</tbody>', parts[i], re.DOTALL)
        if tbody_match:
            new_tbody = replace_table_body(tbody_match)
            parts[i] = parts[i].replace(tbody_match.group(0), new_tbody)

new_html = '<table class="bench-table">'.join(parts)

with open('/ov2/xiangan/sparrow/index.html', 'w', encoding='utf-8') as f:
    f.write(new_html)
