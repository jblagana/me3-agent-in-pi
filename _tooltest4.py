import json, sys, urllib.request, urllib.error
sys.path.insert(0,'/home/jan/me3')
import me2_ui_mcp as ui, web_tools as web, spotify_tools as sp
from pipecat.adapters.schemas.function_schema import FunctionSchema
def norm(p):
    out={}
    for k,v in (p or {}).items():
        out[k] = ({'type':v} if isinstance(v,str) else (v if isinstance(v,(dict,bool)) else {'type':'string'}))
    return out
tools=[]
for spec in (ui.TOOL_SPECS, sp.TOOL_SPECS, web.TOOL_SPECS):
    for (n,d,p) in spec:
        fs=FunctionSchema(n,d,norm(p),list(p.keys()))
        dd=fs.to_default_dict()
        tools.append({'type':'function','function':dd})
print('tools:', len(tools))
print('spotify_play_search:', json.dumps(next(t['function'] for t in tools if t['function']['name']=='spotify_play_search')['parameters']))
KEY='***REMOVED-GROQ-KEY***'
body={'model':'qwen/qwen3.8-27b','messages':[{'role':'user','content':'Pause the music please.'}],'tools':tools,'max_tokens':40,'temperature':0.4}
req=urllib.request.Request('https://api.groq.com/openai/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json','User-Agent':'me3-agent/1.0','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=60) as r: d=json.loads(r.read())
    ch=d['choices'][0]; print('LLM OK. content:', repr(ch['message'].get('content'))); print('  tool_calls:', json.dumps(ch['message'].get('tool_calls')))
except urllib.error.HTTPError as e:
    print('LLM HTTP', e.code); print(e.read().decode()[:500])
