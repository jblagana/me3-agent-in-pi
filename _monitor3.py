import json, time, urllib.request
last={}; t0=time.time(); DUR=55
print('MONITOR3 START. Talk to Boots (e.g. set a timer, then say something).\n', flush=True)
while time.time()-t0<DUR:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8330/ui_state',timeout=1) as r: m=json.load(r)['me3']
        s={k:m.get(k) for k in ('phase','heard','thinking','response','tool')}
        if s!=last:
            ph=s.get('phase'); th=(s.get('thinking') or ''); resp=(s.get('response') or ''); hd=(s.get('heard') or ''); tool=s.get('tool') or ''
            L='[%s] ph=%s'%(time.strftime('%H:%M:%S'),ph)
            if hd: L+=' heard=%r'%hd[:40]
            if th: L+=' thinking=%r'%th[:45]
            if resp: L+=' resp=%r'%resp[:45]
            if tool: L+=' tool=%s'%tool
            print(L,flush=True); last=s
    except Exception: pass
    time.sleep(0.3)
print('\nMONITOR3 END.',flush=True)
