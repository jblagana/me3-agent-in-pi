import base64, time, sys
sys.path.insert(0,'/home/jan/me3')
from remote_voice import Qwen2AudioSTTService, ASR_PROVIDER, ASR_MODEL, GROQ_BASE_URL
print('ASR_PROVIDER=%s MODEL=%s URL=%s' % (ASR_PROVIDER, ASR_MODEL, GROQ_BASE_URL))
wav=open('/tmp/mictest.wav','rb').read()
t0=time.time()
txt=Qwen2AudioSTTService._transcribe_sync(base64.b64encode(wav).decode())
print('REAL _transcribe_sync (%.2fs): %r' % (time.time()-t0, txt))
