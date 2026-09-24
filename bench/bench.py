#!/usr/bin/env python3
"""Repeatable API benchmarks. No user prompts or credentials are saved."""
import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

BASE = os.environ.get('FLASH_BENCH_BASE', 'http://127.0.0.1:30000')
MODEL = 'qwen3.8-flash-next'
ROOT = pathlib.Path(__file__).resolve().parent
CACHE_SALT = 'flash-tuning-initial'


def api(path, body=None, timeout=240):
    req = urllib.request.Request(BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def metrics():
    with api('/metrics') as response:
        text = response.read().decode()
    values = {}
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        name = line.split('{')[0].split()[0]
        if name.endswith('_created'):
            continue
        try:
            values[name] = values.get(name, 0) + float(line.rsplit(' ', 1)[1])
        except ValueError:
            pass
    return values


def idle():
    for _ in range(60):
        m = metrics()
        if m.get('vllm:num_requests_running', 0) == 0 and m.get('vllm:num_requests_waiting', 0) == 0:
            time.sleep(0.5)
            return
        time.sleep(2)
    raise RuntimeError('API not idle; refusing to change test settings')


def stream(name, prompt, max_tokens=512, thinking=False, first_event=None):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'seed': 42, 'max_tokens': max_tokens,
            'chat_template_kwargs': {'enable_thinking': thinking},
            'cache_salt': CACHE_SALT,
            'stream': True, 'stream_options': {'include_usage': True}}
    start = time.perf_counter()
    arrivals, content, reasoning, usage, finish = [], [], [], {}, None
    with api('/v1/chat/completions', body) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                break
            chunk = json.loads(raw)
            if chunk.get('error'):
                raise RuntimeError(chunk['error'])
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
                if text or thought:
                    arrivals.append(time.perf_counter() - start)
                    if first_event:
                        first_event.set()
                content.append(text)
                reasoning.append(thought)
                finish = choice.get('finish_reason') or finish
    elapsed = time.perf_counter() - start
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    n = usage.get('completion_tokens', 0)
    ttft = arrivals[0] if arrivals else elapsed
    return {'name': name, 'seconds': elapsed, 'ttft': ttft,
            'completion_tokens': n, 'prompt_tokens': usage.get('prompt_tokens'),
            'decode_tps_approx': max(0, n - 1) / max(0.001, elapsed - ttft),
            'end_to_end_tps': n / elapsed,
            'p95_stream_gap': sorted(gaps)[int(.95 * (len(gaps) - 1))] if gaps else 0,
            'max_stream_gap': max(gaps, default=0), 'finish_reason': finish,
            'content': ''.join(content), 'reasoning': ''.join(reasoning), 'usage': usage}


CODE_CASES = [
    ('merge_intervals', 'Write only Python code defining merge_intervals(intervals). '
     'Input is a list of [start, end] integer pairs with start <= end. Return a new '
     'list sorted by start, merging overlapping or touching closed intervals. '
     'Handle empty input and do not mutate the input. Use no imports or type annotations. '
     'Include a short docstring, but no example calls or extra text.'),
    ('is_balanced', 'Write only Python code defining is_balanced(text). Return whether '
     'parentheses, square brackets and curly braces are properly nested and matched. '
     'Ignore all other characters. Handle an empty string. Use no imports or type '
     'annotations. Include a short docstring, but no example calls or extra text.'),
    ('dedupe_records', 'Write only Python code defining dedupe_records(records). Each '
     'record is a dictionary with an id key. Keep the last record for each id, while '
     'preserving the order in which distinct ids first appeared. Return a list, handle '
     'empty input, and do not mutate input. Use no imports or type annotations. '
     'Include a short docstring, but no example calls or extra text.'),
]


def validate_code(name, output):
    matches = re.findall(r'```(?:python)?\s*\n(.*?)```', output, re.S)
    source = matches[0] if matches else output.strip()
    try:
        tree = ast.parse(source)
        # Restrict generated code before evaluating simple pure-function examples.
        allowed_calls = {'sorted', 'list', 'dict', 'set', 'tuple', 'len', 'range',
                         'enumerate', 'zip', 'min', 'max', 'any', 'all', 'reversed', 'bool'}
        allowed_attrs = {'append', 'pop', 'get', 'items', 'values', 'keys', 'copy', 'sort', 'add'}
        if any(not isinstance(n, ast.FunctionDef) for n in tree.body):
            raise ValueError('Only function definitions are permitted')
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef)):
                raise ValueError('Disallowed syntax')
            if isinstance(node, ast.Name) and '__' in node.id:
                raise ValueError('Dunder name')
            if isinstance(node, ast.Attribute) and node.attr not in allowed_attrs:
                raise ValueError('Disallowed attribute')
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in allowed_calls:
                    raise ValueError('Disallowed call')
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    raise ValueError('Indirect call')
        # The separate process has CPU/time/memory limits and receives only validated code.
        tests = {
            'merge_intervals': "f=scope['merge_intervals']; x=[[5,7],[1,3],[3,4],[9,9]]; assert f(x)==[[1,4],[5,7],[9,9]]; assert x==[[5,7],[1,3],[3,4],[9,9]]; assert f([])==[]; assert f([[1,9],[2,3]])==[[1,9]]; assert f([[0,0]])==[[0,0]]",
            'is_balanced': "f=scope['is_balanced']; assert all(f(s) for s in ['', 'abc', '{a[(b)]}', '()[]{}']); assert all(not f(s) for s in ['(', ')', '([)]', '{[}'])",
            'dedupe_records': "f=scope['dedupe_records']; x=[{'id':2,'v':'a'},{'id':1,'v':'b'},{'id':2,'v':'c'}]; assert f(x)==[{'id':2,'v':'c'},{'id':1,'v':'b'}]; assert len(x)==3; assert f([])==[]; assert f([{'id':'x'},{'id':'x'}])==[{'id':'x'}]",
        }
        runner = ("import resource\nresource.setrlimit(resource.RLIMIT_CPU,(2,2))\n"
                  "resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456))\n"
                  "scope={'__builtins__':{k:__builtins__.__dict__[k] for k in " + repr(sorted(allowed_calls)) + "}}\n"
                  "exec(compile(" + repr(source) + ",'<model>','exec'),scope)\n" + tests[name])
        result = subprocess.run([sys.executable, '-I', '-c', runner], capture_output=True, text=True, timeout=5)
        return {'pass': result.returncode == 0, 'detail': result.stderr[-500:]}
    except Exception as error:
        return {'pass': False, 'detail': str(error)}


def retrieval_prompt(salt='quality', rows=600):
    records = '\n'.join(f'Record {i:04d}: value_{i:04d}; note {salt}.' for i in range(rows))
    return records + '\nReturn only the value in Record 0173, without explanation.'


def run_suite(label, repeats=2, include_overlap=True):
    global CACHE_SALT
    CACHE_SALT = 'flash-tuning-' + label
    idle()
    before = metrics()
    out = {'label': label, 'started': time.time(), 'requests': []}
    destination = ROOT / 'results' / (label + '.json')
    destination.parent.mkdir(exist_ok=True)
    def save(result):
        out['requests'].append(result)
        destination.write_text(json.dumps(out, indent=2))
        print(json.dumps({k: result[k] for k in ['name', 'seconds', 'ttft', 'completion_tokens', 'decode_tps_approx', 'quality'] if k in result}), flush=True)
    for repeat in range(repeats):
        for name, prompt in CODE_CASES:
            result = stream(f'{name}_{repeat}', prompt)
            result['quality'] = validate_code(name, result['content'])
            save(result)
    for i in range(3):
        result = stream(f'retrieval_{i}', retrieval_prompt(), max_tokens=32)
        result['quality'] = {'pass': result['content'].strip().strip('`') == 'value_0173'}
        save(result)
    # Keep reasoning enabled on a fixed arithmetic check; other tests explicitly disable it.
    result = stream('reasoning_math', 'Compute 12345 * 6789. Give the exact integer result.', max_tokens=512, thinking=True)
    result['quality'] = {'pass': str(12345*6789) in result['content'].replace(',', '')}
    save(result)
    if include_overlap:
        for repeat in range(2):
            started = threading.Event()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                active = pool.submit(stream, f'overlap_code_{repeat}', CODE_CASES[0][1], 512, False, started)
                if not started.wait(60):
                    raise RuntimeError('Decode request failed to start')
                other = pool.submit(stream, f'overlap_prefill_{repeat}', retrieval_prompt(f'overlap{repeat}', 1100), 32)
                a, b = active.result(), other.result()
                a['quality'] = validate_code('merge_intervals', a['content'])
                b['quality'] = {'pass': 'value_0173' in b['content']}
                save(a)
                save(b)
    after = metrics()
    out['metric_delta'] = {key: after.get(key, 0)-value for key, value in before.items() if key.endswith(('_total', '_sum', '_count'))}
    code = [r for r in out['requests'] if any(r['name'].startswith(n+'_') for n,_ in CODE_CASES)]
    out['summary'] = {'all_quality_pass': all(r['quality']['pass'] for r in out['requests']),
        'median_code_decode_tps': statistics.median(r['decode_tps_approx'] for r in code),
        'median_code_seconds': statistics.median(r['seconds'] for r in code),
        'median_code_ttft': statistics.median(r['ttft'] for r in code),
        'retrieval_answers_match': len({r['content'] for r in out['requests'] if r['name'].startswith('retrieval_')})==1,
        'overlap_max_gaps': [r['max_stream_gap'] for r in out['requests'] if r['name'].startswith('overlap_code')],
        'overlap_code_seconds': [r['seconds'] for r in out['requests'] if r['name'].startswith('overlap_code')]}
    destination.write_text(json.dumps(out, indent=2))
    print('SUMMARY '+json.dumps(out['summary']), flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--no-overlap', action='store_true')
    args = parser.parse_args()
    run_suite(args.label, args.repeats, not args.no_overlap)
