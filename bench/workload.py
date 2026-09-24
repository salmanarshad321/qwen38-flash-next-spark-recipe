"""Matched scheduler workloads; real functional checks and full traffic audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from pathlib import Path
import statistics
import threading
import time

import bench

ROOT = Path(__file__).resolve().parent


def run(label, repeats=3, skip_concurrency=False):
    bench.ROOT = ROOT
    bench.idle()
    # Exclude warm-up from timing and token-accounting windows.
    bench.CACHE_SALT = label+'-warmup'
    for name,prompt in bench.CODE_CASES:
        warm = bench.stream(name, prompt)
        assert bench.validate_code(name,warm['content'])['pass']
    warm = bench.stream('prefill-warmup',bench.retrieval_prompt('scheduler-probe',1000),32)
    assert warm['content'].strip().strip('`') == 'value_0173'
    bench.idle()
    before = bench.metrics()
    result = {'label':label,'repeats':repeats,'started':time.time(), 'requests':[], 'concurrency':[], 'mixed':[]}
    dest = ROOT/'results'/(label+'.json')
    dest.parent.mkdir(exist_ok=True)

    def save(item):
        result['requests'].append(item)
        dest.write_text(json.dumps(result,indent=2)+'\n')

    def code(name, case=0, event=None, validate=True):
        key,prompt = bench.CODE_CASES[case]
        r = bench.stream(name,prompt,512,False,event)
        r['_stream_finished_at'] = time.perf_counter()
        if validate:
            r['quality'] = bench.validate_code(key,r['content'])
            r['quality']['pass'] &= r['finish_reason'] != 'length'
        return r

    for repeat in range(2):
        bench.CACHE_SALT = label+'-core'
        for case,(name,_) in enumerate(bench.CODE_CASES):
            r=code(f'code-{name}-{repeat}',case)
            save(r)
    r=bench.stream('reasoning','Compute 12345 * 6789. Give the exact integer result.',512,True)
    integers=re.findall(r'(?<![\d])[-+]?\d+(?![\d])',r['content'].replace(',',''))
    r['quality']={'pass':str(12345*6789) in integers and r['finish_reason']=='stop'}
    save(r)
    print('CORE '+json.dumps({'all_pass':all(r['quality']['pass'] for r in result['requests'])}),flush=True)

    prompt=bench.retrieval_prompt('scheduler-probe',1000)
    def retrieve(name, prompt=prompt, expected='value_0173'):
        r=bench.stream(name,prompt,32)
        r['quality']={'pass':r['content'].strip().strip('`')==expected}
        save(r)
        return r
    cold=[]; warm=[]; repeated=[]
    for rep in range(repeats):
        bench.CACHE_SALT=f'{label}-cold-{rep}'
        cold.append(retrieve(f'cold-{rep}'))
    bench.CACHE_SALT=label+'-shared'
    retrieve('shared-populate')
    for rep in range(repeats):
        changed=prompt.replace('Return only the value in Record 0173',f'Return only the value in Record {429+rep:04d}')
        warm.append(retrieve(f'warm-{rep}',changed,f'value_{429+rep:04d}'))
    for rep in range(repeats):
        repeated.append(retrieve(f'repeat-{rep}'))
    result['prefill']={'cold':[r['ttft'] for r in cold], 'warm':[r['ttft'] for r in warm],
                       'repeated':[r['ttft'] for r in repeated],'prompt_tokens':cold[0]['prompt_tokens']}
    print('PREFILL '+json.dumps(result['prefill']),flush=True)

    if not skip_concurrency:
        for rep in range(2):
            for n in [1,4,8]:
                bench.CACHE_SALT=f'{label}-concurrent-{rep}-{n}'
                start=time.perf_counter()
                with ThreadPoolExecutor(max_workers=n) as pool:
                    futures=[pool.submit(code,f'concurrent-{rep}-{n}-{i}',0,None,False) for i in range(n)]
                    responses=[f.result() for f in as_completed(futures)]
                elapsed=max(r['_stream_finished_at'] for r in responses)-start
                for r in responses:
                    r['quality']=bench.validate_code('merge_intervals',r['content'])
                    r['quality']['pass'] &= r['finish_reason'] != 'length'
                    save(r)
                batch={'repeat':rep,'concurrency':n,'seconds':elapsed,
                       'tokens':sum(r['completion_tokens'] for r in responses),
                       'aggregate_tps':sum(r['completion_tokens'] for r in responses)/elapsed,
                       'median_request_tps':statistics.median(r['decode_tps_approx'] for r in responses),
                       'max_ttft':max(r['ttft'] for r in responses),
                       'all_pass':all(r['quality']['pass'] for r in responses)}
                result['concurrency'].append(batch)
                print('CONCURRENCY '+json.dumps(batch),flush=True)

    for rep in range(repeats):
        bench.CACHE_SALT=f'{label}-mixed-{rep}'
        event=threading.Event()
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            active=pool.submit(code,f'mixed-code-{rep}',0,event)
            if not event.wait(60):
                raise RuntimeError('Decode did not start')
            launch=time.perf_counter()-start
            other=pool.submit(bench.stream,f'mixed-prefill-{rep}',bench.retrieval_prompt('scheduler-mixed',1100),32)
            a,b=active.result(),other.result()
        b['quality']={'pass':b['content'].strip().strip('`')=='value_0173'}
        save(a); save(b)
        batch={'repeat':rep,'prefill_launch_after_code_start':launch,'code_seconds':a['seconds'],
               'code_max_stream_gap':a['max_stream_gap'],'code_p95_stream_gap':a['p95_stream_gap'],
               'prefill_ttft':b['ttft'],'prefill_tokens':b['prompt_tokens'],
               'all_pass':a['quality']['pass'] and b['quality']['pass']}
        result['mixed'].append(batch)
        print('MIXED '+json.dumps(batch),flush=True)
    bench.idle()
    after=bench.metrics()
    delta={k:after.get(k,0)-v for k,v in before.items() if k.endswith(('_total','_sum','_count'))}
    result['metric_delta']=delta
    expected=sum(r['completion_tokens'] for r in result['requests'])
    result['traffic_clean']=(delta.get('vllm:request_success_total')==len(result['requests']) and delta.get('vllm:generation_tokens_total')==expected)
    result['all_pass']=all(r['quality']['pass'] for r in result['requests'])
    codes=[r for r in result['requests'] if r['name'].startswith('code-')]
    result['summary']={'all_pass':result['all_pass'],'traffic_clean':result['traffic_clean'],
                       'requests':len(result['requests']),'tokens':expected,
                       'code_tps':statistics.median(r['decode_tps_approx'] for r in codes),
                       'cold_ttft':statistics.median(result['prefill']['cold']),
                       'warm_ttft':statistics.median(result['prefill']['warm']),
                       'mixed_max_gap_median':statistics.median(b['code_max_stream_gap'] for b in result['mixed']),
                       'mixed_prefill_ttft':statistics.median(b['prefill_ttft'] for b in result['mixed']),
                       'preemptions':delta.get('vllm:num_preemptions_total')}
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY '+json.dumps(result['summary']),flush=True)
    if not result['all_pass'] or not result['traffic_clean']:
        raise RuntimeError('Functional checks or traffic accounting failed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('label')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--skip-concurrency',action='store_true')
    a=p.parse_args()
    run(a.label,a.repeats,a.skip_concurrency)
