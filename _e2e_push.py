# Drive the REAL deployed agent.py push logic through a real FunctionCallsStartedFrame
# and LLMTextFrames, then read back /ui_state to prove agent->UI->state works.
import json, time, urllib.request, importlib.util, asyncio
spec=importlib.util.spec_from_file_location('agentmod','/home/jan/me3/agent.py')
# agent.py runs pipeline on import? No -- it has if __name__=='__main__' guard. Import is safe.
agent=importlib.util.module_from_spec(spec); spec.loader.exec_module(agent)
from pipecat.frames.frames import FunctionCallsStartedFrame, FunctionCallFromLLM, LLMTextFrame
si=agent.StatusIndicator(state={'bot_speaking':False})
# 1) Simulate a tool call (what the LLM aggregator pushes)
fc=[FunctionCallFromLLM(function_name='spotify_play_search',tool_call_id='t1',arguments={'query':'upbeat pop'},context=None)]
asyncio.get_event_loop().run_until_complete(si.process_frame(FunctionCallsStartedFrame(function_calls=fc), None))
# 2) Simulate streaming LLM text (thinking/reply tokens)
agent._RECENT_TEXT['transcript']='play some upbeat pop'
for tok in ['Sure,',' playing',' upbeat',' pop']:  # noqa
    asyncio.get_event_loop().run_until_complete(si.process_frame(LLMTextFrame(tok), None))
time.sleep(1.5)  # let fire-and-forget POST threads complete
def ui():
    with urllib.request.urlopen('http://127.0.0.1:8330/ui_state',timeout=2) as r:
        return json.load(r)['ui_state']['me3']
m=ui()
print('UI.me3.tool      =', repr(m.get('tool')))
print('UI.me3.tool_args =', repr(m.get('tool_args')))
print('UI.me3.thinking  =', repr(m.get('thinking')))
print('UI.me3.response  =', repr(m.get('response')))
print('UI.me3.phase     =', repr(m.get('phase')))
ok = (m.get('tool')=='spotify_play_search' and m.get('tool_args') and 'upbeat pop' in m.get('tool_args') and m.get('thinking'))
print('E2E_TOOL_STREAM:', 'PASS' if ok else 'FAIL')
