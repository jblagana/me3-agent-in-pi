import base64, time, sys
sys.path.insert(0,'/home/jan/me3')
from remote_voice import Qwen2AudioSTTService, ASR_PROVIDER, ASR_MODEL
svc=Qwen2AudioSTTService()
wav=open('/tmp/mictest.wav','rb').read()
t0=time.time()
txt=svc._transcribe_sync(base64.b64encode(wav).decode())
print('REAL _transcribe_sync (%.2fs): %r  [provider=%s model=%s]' % (time.time()-t0, txt, ASR_PROVIDER, ASR_MODEL))
