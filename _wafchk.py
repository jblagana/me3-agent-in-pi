import urllib.request, urllib.error, json
def probe():
    KEY='***REMOVED-GROQ-KEY***'
    body={'model':'llama-3.1-8b-instant','messages':[{'role':'user','content':'ping'}],'max_tokens':3}
    req=urllib.request.Request('https://api.groq.com/openai/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json','User-Agent':'me3-agent/1.0','Accept':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=30) as r:
            return r.status, r.read().decode()[:120]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:220]
    except Exception as e:
        return -1, repr(e)
code,body=probe()
print('CODE', code)
print('BODY', body)
print('CLOUDFLARE' if 'cloudflare' in body.lower() or 'cf-' in body.lower() or 'ray id' in body.lower() else 'NOT-CLF')
