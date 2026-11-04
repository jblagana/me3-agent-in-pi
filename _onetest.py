import json, urllib.request, urllib.error, base64, time, sys
sys.path.insert(0,'/home/jan/me3')
KEY='***REMOVED-GROQ-KEY***'
H={'Authorization':'Bearer '+KEY,'User-Agent':'me3-agent/1.0','Accept':'application/json'}
from remote_voice import Qwen2AudioSTTService
wav=open('/tmp/mictest.wav','rb').read()
t0=time.time()
try:
    txt=Qwen2AudioSTTService._transcribe_groq(base64.b64encode(wav).decode())
    print('ASR (%.2fs): %r' % (time.time()-t0, txt))
except urllib.error.HTTPError as e:
    print('ASR HTTP', e.code, 'body:', e.read().decode()[:200])
t0=time.time()
body={'model':'qwen/qwen3.8-27b','messages':[{'role':'user','content':'Say exactly: Groq LLM works.'}], 'max_tokens':30,'temperature':0.0}
req=urllib.request.Request('https://api.groq.com/openai/v1/chat/completions',data=json.dumps(body).encode(),headers=H)
try:
    with urllib.request.urlopen(req,timeout=60) as r: llm=json.loads(r.read())['choices'][0]['message']['content']
    print('LLM (%.2fs): %r' % (time.time()-t0, llm))
except urllib.error.HTTPError as e:
    print('LLM HTTP', e.code, 'body:', e.read().decode()[:200])
