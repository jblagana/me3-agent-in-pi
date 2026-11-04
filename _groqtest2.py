import json, urllib.request, uuid, time, sys
KEY='***REMOVED-GROQ-KEY***'
GROQ='https://api.groq.com/openai/v1'
wav=open('/tmp/mictest.wav','rb').read()
print('WAV bytes:', len(wav))
# ASR
b=bytearray(); boundary='----me3'+uuid.uuid4().hex
b+=('--'+boundary+'\r\n').encode()
b+=b'Content-Disposition: form-data; name= file; filename=a.wav\r\n'
b+=b'Content-Type: audio/wav\r\n\r\n'
b+=wav; b+=b'\r\n'
b+=('--'+boundary+'\r\n').encode()
b+=b'Content-Disposition: form-data; name=model\r\n\r\n'
b+=b'whisper-large-v3-turbo'; b+=b'\r\n'
req=urllib.request.Request(GROQ+'/audio/transcriptions',data=bytes(b),headers={'Authorization':'Bearer '+KEY,'Content-Type':'multipart/form-data; boundary='+boundary})
try:
    t0=time.time()
    with urllib.request.urlopen(req,timeout=60) as r: txt=json.loads(r.read()).get('text','')
    print('ASR (%.2fs): %r' % (time.time()-t0, txt))
except Exception as e:
    print('ASR FAILED:', repr(e)); sys.exit(1)
# LLM
body={'model':'llama-3.3-70b-versatile','messages':[{'role':'user','content':'Say exactly: Groq LLM is working.'}], 'max_tokens':30,'temperature':0.0}
req=urllib.request.Request(GROQ+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
try:
    t0=time.time()
    with urllib.request.urlopen(req,timeout=60) as r: llm=json.loads(r.read())['choices'][0]['message']['content']
    print('LLM (%.2fs): %r' % (time.time()-t0, llm))
except Exception as e:
    print('LLM FAILED:', repr(e)); sys.exit(2)
print('GROQ_TEST_OK')
