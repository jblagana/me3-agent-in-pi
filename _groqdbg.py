import urllib.request, urllib.error, json
KEY='***REMOVED-GROQ-KEY***'
GROQ='https://api.groq.com/openai/v1'
body={'model':'llama-3.3-70b-versatile','messages':[{'role':'user','content':'ping'}],'max_tokens':5}
req=urllib.request.Request(GROQ+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json','User-Agent':'python-urllib/3.13'})
try:
    with urllib.request.urlopen(req,timeout=60) as r: print('STATUS', r.status); print(r.read().decode()[:500])
except urllib.error.HTTPError as e:
    print('HTTP', e.code, e.reason); print('BODY:', e.read().decode()[:500])
except Exception as e:
    print('ERR', repr(e))
