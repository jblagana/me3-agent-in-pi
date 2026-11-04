import json, time, urllib.request
last = {}
t0 = time.time(); DUR = 60
print('MONITOR2 START. Talk to Boots (up to %ds)\n' % DUR, flush=True)
while time.time() - t0 < DUR:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8330/ui_state', timeout=1) as r:
            m = json.load(r)['me3']
        snap = {k: m.get(k) for k in ('phase','heard','thinking','response','tool')}
        if snap != last:
            ph = snap.get('phase'); th=(snap.get('thinking') or '')
            resp=(snap.get('response') or ''); heard=(snap.get('heard') or '')
            tool=snap.get('tool') or ''
            line='[%s] ph=%s' % (time.strftime('%H:%M:%S'), ph)
            if heard: line += ' heard=%r' % heard[:45]
            if th: line += ' thinking=%r' % th[:55]
            if resp: line += ' resp=%r' % resp[:55]
            if tool: line += ' tool=%s' % tool
            print(line, flush=True)
            last=snap
    except Exception: pass
    time.sleep(0.3)
print('\nMONITOR2 END.', flush=True)
