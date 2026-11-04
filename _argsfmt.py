import sys; sys.path.insert(0,'/home/jan/me3')
from pipecat.frames.frames import FunctionCallsStartedFrame, FunctionCallFromLLM
# Replicate agent.py's tool-args formatting exactly
fcs = [FunctionCallFromLLM(function_name='spotify_play_search', tool_call_id='x', arguments={'query':'upbeat pop'}, context=None)]
frame = FunctionCallsStartedFrame(function_calls=fcs)
names=[fc.function_name for fc in frame.function_calls]
if len(fcs)==1:
    args=', '.join(f'{k}={v!r}' for k,v in (fcs[0].arguments or {}).items()) or None
else:
    args='; '.join(f'{fc.function_name}({fc.arguments})' for fc in fcs) or None
print('tool:', ', '.join(names))
print('tool_args:', repr(args))
