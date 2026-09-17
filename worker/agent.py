#!/usr/bin/env python3
import os, json, subprocess, hashlib
from pathlib import Path

PREFERRED=['/generate','/chat','/predict','/respond','/infer','/run']
def run(cmd, timeout=240): return subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
def payload_for(spec,prompt):
    payload={}; prompt_set=False
    for p in spec.get('parameters',[]):
        name=p.get('name',''); lname=name.lower(); required=bool(p.get('required',False)); default=p.get('default'); typ=(p.get('type') or {}).get('type')
        if lname in {'message','prompt','text','query','input','instruction','user_message'}: payload[name]=prompt; prompt_set=True
        elif lname in {'chat_history','history','messages'}: payload[name]=[]
        elif lname in {'max_new_tokens','max_tokens','maximum_new_tokens'}: payload[name]=700
        elif lname=='temperature': payload[name]=0.3
        elif lname=='top_p': payload[name]=0.9
        elif lname=='top_k': payload[name]=40
        elif lname in {'repetition_penalty','repeat_penalty'}: payload[name]=1.1
        elif lname in {'system','system_prompt'}: payload[name]='Evidence-bounded invention and discovery analysis. IDEA!=INVENTION; PATENTABILITY!=FTO; CLAIM<=EVIDENCE.'
        elif required and default is None:
            if typ=='string' and not prompt_set: payload[name]=prompt; prompt_set=True
            else: return None
    return payload if prompt_set else None
def extract(raw):
    raw=raw.strip()
    try:
        obj=json.loads(raw)
        if isinstance(obj,dict):
            for k in ('Response','response','text','output','message'):
                if isinstance(obj.get(k),str): return obj[k].strip()
    except Exception: pass
    return raw
def invoke(space,prompt):
    info=run(['hf-gradio','info',space],120)
    if info.returncode!=0: return False,'',{'stage':'info','error':(info.stderr or info.stdout)[-1200:]}
    try: api=json.loads(info.stdout)
    except Exception as e: return False,'',{'stage':'decode','error':repr(e)}
    endpoints=list(api.items()); endpoints.sort(key=lambda kv:(PREFERRED.index(kv[0]) if kv[0] in PREFERRED else 99,kv[0]))
    errors=[]
    for endpoint,spec in endpoints:
        payload=payload_for(spec,prompt)
        if payload is None: continue
        pred=run(['hf-gradio','predict',space,endpoint,json.dumps(payload,ensure_ascii=False)],240)
        if pred.returncode==0 and (pred.stdout or '').strip():
            text=extract(pred.stdout)
            if text: return True,text,{'stage':'predict','endpoint':endpoint,'sha256':hashlib.sha256(text.encode()).hexdigest()}
        errors.append((pred.stderr or pred.stdout)[-700:])
    return False,'',{'stage':'predict','error':' | '.join(errors[-3:]) or 'No compatible endpoint'}

role=os.environ['ROLE']; space=os.environ.get('MODEL','huggingface-projects/llama-3.2-3B-Instruct'); focus=os.environ.get('FOCUS','invention discovery')
prompt=f'''You are role {role} in CEREBRON Omega Farm 32 Invention Discovery.\nFocus: {focus}.\nGenerate or audit invention/discovery candidates with strict evidence discipline. Separate ESTABLISHED / DERIVED / HYPOTHESIS / SPECULATIVE. For each candidate state problem, mechanism, why it may work, assumptions, dependencies, novelty risk, feasibility, manufacturability or implementability, cost drivers, safety risks, critical unknowns, falsification test, minimum decisive experiment, transfer limits and failure modes. Do not claim patentability or freedom-to-operate without evidence. Do not call simulations tests. Unknown remains unknown.'''
ok,text,meta=invoke(space,prompt)
out={'role':role,'model':space,'focus':focus,'provider':'huggingface-space-zerogpu','inference_success':bool(ok),'status':'UNREVIEWED_EXTERNAL_AGENT_OUTPUT' if ok else 'EXTERNAL_INFERENCE_FAILED','output':text if ok else None,'error':None if ok else meta.get('error'),'meta':meta}
Path('results').mkdir(exist_ok=True); Path(f'results/{role}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'role':role,'model':space,'status':out['status'],'inference_success':out['inference_success']},ensure_ascii=False))
