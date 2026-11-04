import json, time, urllib.request

# Reproduce EXACTLY what the deployed agent.py _me3_push_now() posts after a
# FunctionCallsStartedFrame (same body construction), then read back /ui_state.me3
# to prove the UI server stores + exposes tool + tool_args + streaming thinking.
phase = 'tool'
body = {
    'phase': phase,
    'heard': 'play some upbeat pop',
    'thinking': 'Sure, playing upbeat pop',
    'response': 'Sure, playing upbeat pop',
    'tool': 'spotify_play_search',
    'tool_args': "query='upbeat pop'",
    'mic': False,
    'active': True,
}


def post(b):
    req = urllib.request.Request(
        'http://127.0.0.1:8330/me3', data=json.dumps(b).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=2) as r:
        r.read()


post(body)
time.sleep(0.5)
with urllib.request.urlopen('http://127.0.0.1:8330/ui_state', timeout=2) as r:
    m = json.load(r)['me3']

print('POST tool      =', repr(body['tool']))
print('POST tool_args =', repr(body['tool_args']))
print('--- read back /ui_state.me3 ---')
print('UI tool        =', repr(m.get('tool')))
print('UI tool_args   =', repr(m.get('tool_args')))
print('UI thinking    =', repr(m.get('thinking')))
print('UI phase       =', repr(m.get('phase')))
ok = (m.get('tool') == 'spotify_play_search'
      and m.get('tool_args') == "query='upbeat pop'"
      and m.get('thinking'))
print('UI_TOOL_ARGS_ROUNDTRIP:', 'PASS' if ok else 'FAIL')

# Reset the live UI back to listening so it is not stuck showing a fake tool.
post({'phase': 'listening', 'heard': '', 'thinking': '', 'response': '',
      'tool': '', 'tool_args': '', 'mic': True, 'active': True})
print('reset to listening')
