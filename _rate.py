import json,urllib.request
with urllib.request.urlopen('http://127.0.0.1:8330/ui_state',timeout=2) as r: m=json.load(r)['me3']
print('phase=%s active=%s' % (m.get('phase'), m.get('active')))
