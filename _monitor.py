import json, time, urllib.request, sys
last = {}
t0 = time.time()
DUR = 60
print('MONITOR START. Talk to Boots now (up to %ds)\n' % DUR, flush=True)
while time.time() - t0 < DUR:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8330/ui_state', timeout=1) as r:
            m = json.load(r)['me3']
        snap = {k: m.get(k) for k in ('phase','heard','thinking','response','tool','tool_args')}
        # only log when something meaningful changed
        if snap != last:
            ph = snap.get('phase'); th = (snap.get('thinking') or '')
            resp = (snap.get('response') or ''); heard=(snap.get('heard') or '')
            tool = snap.get('tool') or ''; ta = snap.get('tool_args') or ''
            line = '[%s] ph=%s' % (time.strftime('%H:%M:%S'), ph)
            if heard: line += ' heard=%r' % heard[:50]
            if th: line += ' thinking=%r' % th[:60]
            if resp: line += ' resp=%r' % resp[:60]
            if tool: line += ' tool=%s(%s)' % (tool, ta[:40])
            print(line, flush=True)
            last = snap
    except Exception as e:
        pass
    time.sleep(0.3)
print('\nMONITOR END.', flush=True)
