import json, sys, urllib.request, urllib.error
sys.path.insert(0,'/home/jan/me3')
import agent
KEY='***REMOVED-GROQ-KEY***'
schemas=agent._function_schemas()
print('tool count:', len(schemas))
# serialize like pipecat FunctionSchema -> OpenAI tool
openai_tools=[]
for s in schemas:
    params={'type':'object','properties':(s.parameters or {}), 'required':(s.required or [])}
    openai_tools.append({'type':'function','function':{'name':s.name,'description':s.description,'parameters':params}})
print('spotify_play_search params:', json.dumps(next(t['function']['parameters'] for t in openai_tools if t['function']['name']=='spotify_play_search')))
body={'model':'qwen/qwen3.8-27b','messages':[{'role':'user','content':'Pause the music please.'}],'tools':openai_tools,'max_tokens':40,'temperature':0.4}
req=urllib.request.Request('https://api.groq.com/openai/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json','User-Agent':'me3-agent/1.0','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=60) as r: d=json.loads(r.read())
    ch=d['choices'][0]; print('LLM RESPONSE OK:'); print('  content:', ch['message'].get('content'))
    print('  tool_calls:', json.dumps(ch['message'].get('tool_calls')))
except urllib.error.HTTPError as e:
    print('LLM HTTP', e.code); print(e.read().decode()[:400])
