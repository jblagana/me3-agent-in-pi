import urllib.request, urllib.error, json
KEY='***REMOVED-GROQ-KEY***'
def get(url):
    req=urllib.request.Request(url,headers={'Authorization':'Bearer '+KEY,'User-Agent':'me3-agent/1.0','Accept':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=30) as r: return r.status, r.read().decode()
    except urllib.error.HTTPError as e: return e.code, e.read().decode()
    except Exception as e: return -1, repr(e)
c,b=get('https://api.groq.com/openai/v1/models')
print('MODELS', c)
if c==200:
    for m in json.loads(b)['data']:
        print('  ', m['id'])
else:
    print(b[:200])
