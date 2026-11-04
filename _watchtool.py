import json, time, urllib.request
seen=[]
t0=time.time()
while time.time()-t0 < 40:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8330/ui_state',timeout=1) as r:
            d=json.load(r)
        m3=d.get('ui_state',{}).get('me3',{})
        if m3.get('tool') or m3.get('tool_args'):
            print('CATCH tool=%r tool_args=%r phase=%r thinking=%r' % (m3.get('tool'), m3.get('tool_args'), m3.get('phase'), (m3.get('thinking') or '')[:60]))
            seen.append(m3.get('tool'))
    except Exception:
        pass
    time.sleep(0.25)
print('DONE. tool events seen:', len(seen), seen[:10])
