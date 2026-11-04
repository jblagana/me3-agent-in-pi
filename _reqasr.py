import time
try:
    import requests
except Exception as e:
    print('NO REQUESTS:', repr(e)); raise SystemExit
KEY='***REMOVED-GROQ-KEY***'
wav=open('/tmp/mictest.wav','rb').read()
t0=time.time()
r=requests.post('https://api.groq.com/openai/v1/audio/transcriptions',
    headers={'Authorization':'Bearer '+KEY},
    files={'file':('segment.wav', wav, 'audio/wav')},
    data={'model':'whisper-large-v3-turbo'},
    timeout=60)
print('requests ASR status', r.status_code, '(%.2fs)' % (time.time()-t0))
print('text:', r.text[:200])
