import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:8330/ui_state', timeout=2) as r:
    m = json.load(r)['me3']
print('phase     =', m.get('phase'))
print('tool      =', repr(m.get('tool')))
print('tool_args =', repr(m.get('tool_args')))
print('thinking  =', repr((m.get('thinking') or '')[:50]))
print('active    =', m.get('active'))
print('has tool_args key:', 'tool_args' in m)
