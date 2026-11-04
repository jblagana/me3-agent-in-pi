import json, urllib.request, urllib.error, base64, time, sys
sys.path.insert(0,'/home/jan/me3')
KEY='***REMOVED-GROQ-KEY***'
GROQ='https://api.groq.com/openai/v1'
# 1) find a valid LLM model
for m in ['llama-3.3-70b-versatile','llama-3.1-8b-instant','llama-3.3-70b-versatile','openai/gpt-oss-20b']:
    body={'model':m,'messages':[{'role':'user','content':'ping'}],'max_tokens':3}
    req=urllib.request.Request(GROQ+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=60) as r: r.read(); print('LLM OK:', m); valid_llm=m; break
    except urllib.error.HTTPError as e:
        print('LLM no:', m, e.code)
    except Exception as e:
        print('LLM err:', m, repr(e))
# 2) ASR via the REAL remote_voice._transcribe_groq (fixed multipart)
from remote_voice import Qwen2AudioSTTService
wav=open('/tmp/mictest.wav','rb').read()
b64=base64.b64encode(wav).decode()
t0=time.time(); txt=Qwen2AudioSTTService._transcribe_groq(b64)
print('ASR (%.2fs) via remote_voice: %r' % (time.time()-t0, txt))
print('FINAL_OK')
