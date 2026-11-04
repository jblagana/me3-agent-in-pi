import urllib.request, urllib.error, uuid, time
KEY='***REMOVED-GROQ-KEY***'
GROQ='https://api.groq.com/openai/v1'
wav=open('/tmp/mictest.wav','rb').read()
for model in ['whisper-large-v3-turbo','whisper-large-v3','whisper-1']:
    b=bytearray(); boundary='----m'+uuid.uuid4().hex
    b+=('--'+boundary+'\r\n').encode()
    b+=b'Content-Disposition: form-data; name= file; filename=a.wav\r\n'
    b+=b'Content-Type: audio/wav\r\n\r\n'
    b+=wav; b+=b'\r\n'
    b+=('--'+boundary+'\r\n').encode()
    b+=b'Content-Disposition: form-data; name=model\r\n\r\n'
    b+=model.encode(); b+=b'\r\n'
    req=urllib.request.Request(GROQ+'/audio/transcriptions',data=bytes(b),headers={'Authorization':'Bearer '+KEY,'Content-Type':'multipart/form-data; boundary='+boundary,'User-Agent':'python-urllib/3.13'})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=60) as r: txt=r.read().decode()
        print(model, 'OK (%.2fs):' % (time.time()-t0), txt[:150])
    except urllib.error.HTTPError as e:
        print(model, 'HTTP', e.code, e.reason, 'body:', e.read().decode()[:200])
    except Exception as e:
        print(model, 'ERR', repr(e))
