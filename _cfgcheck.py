import os
# replicate agent.py load order: llm.env (setdefault) then groq.env (set)
for path,mode in [('/etc/me3/llm.env','setdefault'),('/etc/me3/groq.env','set')]:
    try:
        for line in open(path):
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line: continue
            k,_,v=line.partition('=')
            (os.environ.setdefault if mode=='setdefault' else (lambda kk,vv: os.environ.__setitem__(kk,vv)))(k.strip(), v.strip())
    except Exception as e: print(path,'ERR',repr(e))
print('LLM_BASE_URL =', os.environ.get('LLM_BASE_URL'))
print('LLM_MODEL    =', os.environ.get('LLM_MODEL'))
print('LLM_API_KEY  =', (os.environ.get('LLM_API_KEY') or '')[:8]+'...')
print('ME3_ASR_PROVIDER =', os.environ.get('ME3_ASR_PROVIDER'))
print('GROQ_BASE_URL =', os.environ.get('GROQ_BASE_URL'))
print('ME3_ASR_MODEL =', os.environ.get('ME3_ASR_MODEL'))
