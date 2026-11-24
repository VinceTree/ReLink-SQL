"""Additional literature adapters, sharing the established feedback runner."""
from types import SimpleNamespace
import json
from .dataset import r

NEW_METHODS=('REFLEXION','SELF_DEBUG')


def checked(task,state):
    public=dict(state,method='EXEC14')
    r.methods.validate(task,public)
    return public


def selfdebug_explanation(task,state,full):
    checked(task,state)
    observed=state['observation']
    if observed['status']=='ok':
        result=dict(status='ok',columns=observed['columns'],rows=observed['rows'][:2])
    else:result=dict(status='error',error=observed.get('error',''))
    prompt=('Summarize the return type of the SQL execution. Explain the meaning of each SQL clause, '
            'the output columns, and the overall meaning of the query.\n\nDatabase schema:\n'+full['schema_text']+
            '\n\nSQL:\n'+state['sql']+'\n\nExecution (at most two result rows):\n'+
            json.dumps(result,ensure_ascii=False,default=str))
    return [dict(role='system',content='You are an expert programming assistant.'),
            dict(role='user',content=prompt)]


def build(task,state,full,stage,auxiliary=None,explanations=None):
    public=checked(task,state)
    if state['method']=='REFLEXION':
        values=dict(schema=full['schema_text'],question=task['question'],evidence=task.get('evidence',''),
                    sql=state['sql'],observation=r.methods.paper.public_observation(state['observation']))
        if stage=='answer':values['feedback']=auxiliary
        template='reflexion_sql_'+('reflect' if stage=='feedback' else 'retry')+'_adapted.txt'
        prompt=r.methods.paper.render(template,values)
        if state['critiques']:
            prompt+='\n\nMemory from previous rejected attempts:\n'+'\n\n'.join(state['critiques'])
        return r.methods.attach([dict(role='user',content=prompt)],public),dict(
            source='Reflexion',sql_domain_adapter=True,memory_items=len(state['critiques']))
    if state['method']=='SELF_DEBUG':
        messages=selfdebug_explanation(task,state,full)
        messages += [dict(role='assistant',content=explanations[task['id']]['text']),
                     dict(role='user',content='Infer the return type of the question.\nQuestion: '+task['question']+
                          '\nPublic domain evidence: '+task.get('evidence',''))]
        if stage=='answer':
            prompt=('Does the SQL match the question? If not, generate the fixed SQL.\n'
                    'User feedback on the current SQL: '+r.methods.NEGATIVE+
                    '\nReturn one complete SQLite SELECT query for the next attempt, without commentary.')
            if public['rejected_sqls'][:-1]:
                prompt+='\nEarlier rejected SQL attempts:\n'+'\n\n'.join(public['rejected_sqls'][:-1])
            messages += [dict(role='assistant',content=auxiliary),dict(role='user',content=prompt)]
        return messages,dict(source='Self-Debugging Appendix E.7',chat_zero_shot=True,
                             execution_preview_rows=2,external_binary_feedback_adapter=True)
    raise ValueError(state['method'])


def round_run(dest,rnd,previous,tasks,full,tables,bank,evaluator,gate='official'):
    if (dest/f'r{rnd}/complete.json').exists():
        return r.read(dest/f'r{rnd}/paths.jsonl')
    method=previous[0]['method']
    if method not in NEW_METHODS:
        return r.round_run(dest,rnd,previous,tasks,full,tables,bank,evaluator,gate)
    explanations={}
    if method=='SELF_DEBUG':
        specs=[]
        for row in previous:
            if row['accepted'] or row.get('stopped'):continue
            state=r.public_state(row,rnd);tid=row['task_id']
            messages=selfdebug_explanation(tasks[tid],state,full[tid])
            sp=r.spec(state,'sql_explanation',messages,tag='full_dev_17')
            sp['request']['max_tokens']=2048;specs.append(sp)
        explanations={x['task_id']:x for x in bank.invoke(dest/f'r{rnd}/explanation',specs)}
        paid=[r.account(dict(x,call_ids=x['call_ids']+explanations[x['task_id']]['request_ids']),bank)
              if x['task_id'] in explanations else dict(x) for x in previous]
        r.save(dest/f'r{rnd}/before_explanation.jsonl',previous)
        r.save(dest/f'r{rnd}/after_explanation.jsonl',paid)
        previous=paid
    def adapter(task,state,context,stage='answer',auxiliary=None):
        return build(task,state,context,stage,auxiliary,explanations)
    proxy=SimpleNamespace(**vars(r.methods))
    proxy.initial_stage=lambda name:'feedback' if name in NEW_METHODS else r.methods.initial_stage(name)
    execute=r.methods.selectors.clone(r.round_run,methods=proxy,build=adapter)
    return execute(dest,rnd,previous,tasks,full,tables,bank,evaluator,gate)
