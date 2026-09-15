#!/usr/bin/env python3
"""Capture vector, attention, and recurrent fixtures from pinned CPU graphs."""
import argparse
import ctypes as c
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = '7620399f58aebfd2196b74021f9581bcf7218cb9'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkout', nargs='?', type=Path, default=ROOT / '.zig-cache/reference/llama.cpp')
checkout = parser.parse_args().checkout.resolve()
if subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
    parser.error('reference checkout must match the pinned revision')
lib = c.CDLL(str(checkout / 'build/bin/libggml-base.dylib'), mode=c.RTLD_GLOBAL)
cpu = c.CDLL(str(checkout / 'build/bin/libggml-cpu.dylib'))

class Init(c.Structure):
    _fields_ = [('mem_size', c.c_size_t), ('mem_buffer', c.c_void_p), ('no_alloc', c.c_bool)]

def function(library, name, result, args):
    fn = getattr(library, name)
    fn.restype, fn.argtypes = result, args
    return fn

p = c.c_void_p
init = function(lib, 'ggml_init', p, [Init])
free = function(lib, 'ggml_free', None, [p])
tensor = function(lib, 'ggml_new_tensor_1d', p, [p, c.c_int, c.c_int64])
data = function(lib, 'ggml_get_data', p, [p])
l2 = function(lib, 'ggml_l2_norm', p, [p, p, c.c_float])
rope = function(lib, 'ggml_rope_multi', p, [p, p, p, p, c.c_int, c.POINTER(c.c_int), c.c_int, c.c_int] + [c.c_float] * 6)
graph = function(lib, 'ggml_new_graph', p, [p])
expand = function(lib, 'ggml_build_forward_expand', None, [p, p])
compute = function(cpu, 'ggml_graph_compute_with_ctx', c.c_int, [p, p, c.c_int])

def evaluate(values, epsilon=None, position=None):
    ctx = init(Init(16 * 1024 * 1024, None, False))
    if not ctx:
        raise RuntimeError('ggml context allocation failed')
    try:
        source = tensor(ctx, 0, len(values))
        raw = (c.c_float * len(values))(*values)
        c.memmove(data(source), raw, c.sizeof(raw))
        if epsilon is not None:
            result = l2(ctx, source, epsilon)
        else:
            positions = tensor(ctx, 26, 4)  # GGML_TYPE_I32
            raw_pos = (c.c_int32 * 4)(position, position, position, 0)
            c.memmove(data(positions), raw_pos, c.sizeof(raw_pos))
            sections = (c.c_int * 4)(11, 11, 10, 0)
            result = rope(ctx, source, positions, None, 64, sections, 40,
                          32768, 10000000, 1, 0, 1, 32, 1)
        g = graph(ctx)
        expand(g, result)
        if compute(ctx, g, 1) != 0:
            raise RuntimeError('CPU graph computation failed')
        return list(c.cast(data(result), c.POINTER(c.c_float * len(values))).contents)
    finally:
        free(ctx)

l2_cases = []
for values, epsilon in [([3, 4], 1e-6), ([3, 4], 10), ([0, 0], 1e-6), ([1e-7, -2e-7], 1e-6)]:
    values = [c.c_float(x).value for x in values]
    epsilon = c.c_float(epsilon).value
    l2_cases.append(dict(input=values, epsilon=epsilon, output=evaluate(values, epsilon=epsilon)))
values = [((i * 17) % 31 - 15) / 8 for i in range(256)]
rope_cases = [dict(position=pos, output=evaluate(values, position=pos)) for pos in [0, 1, 127, 32767]]
target = ROOT / 'inference/src/backends/cpu/fixtures/vector.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(dict(revision=REVISION, l2=l2_cases, rope_input=values,
                                 rope_dimensions=64, rope_base=10000000,
                                 rope_mode=40, rope_sections=[11, 11, 10, 0],
                                 rope=rope_cases), indent=2) + '\n')

tensor3 = function(lib, 'ggml_new_tensor_3d', p, [p, c.c_int, c.c_int64, c.c_int64, c.c_int64])
mul = function(lib, 'ggml_mul_mat', p, [p, p, p])
softmax = function(lib, 'ggml_soft_max_ext', p, [p, p, p, c.c_float, c.c_float])

def attention(case):
    ctx = init(Init(16 * 1024 * 1024, None, False))
    if not ctx:
        raise RuntimeError('ggml context allocation failed')
    try:
        qh, kh = case['query_heads'], case['kv_heads']
        kw, vw, n = case['key_width'], case['value_width'], case['visible_tokens']

        def loaded(shape, values):
            t = tensor3(ctx, 0, *shape)
            raw = (c.c_float * len(values))(*values)
            c.memmove(data(t), raw, c.sizeof(raw))
            return t

        q = loaded((kw, 1, qh), case['queries'])
        # GGML matrix axes differ from the library's token-major KV storage.
        # Only materialize the visible prefix, equivalent to masking the suffix.
        k = loaded((kw, n, kh), [case['keys'][(t * kh + h) * kw + d]
                                 for h in range(kh) for t in range(n) for d in range(kw)])
        v = loaded((n, vw, kh), [case['values'][(t * kh + h) * vw + d]
                                 for h in range(kh) for d in range(vw) for t in range(n)])
        scores = mul(ctx, k, q)
        probabilities = softmax(ctx, scores, None, case['scale'], 0)
        result = mul(ctx, v, probabilities)
        g = graph(ctx)
        expand(g, result)
        if compute(ctx, g, 1) != 0:
            raise RuntimeError('CPU attention graph failed')
        return list(c.cast(data(result), c.POINTER(c.c_float * (qh * vw))).contents)
    finally:
        free(ctx)

cases = []
for heads, kv_heads in [(4, 2), (3, 1)]:
    for visible in [1, 2, 4]:
        case = dict(query_heads=heads, kv_heads=kv_heads, key_width=3,
                    value_width=2, tokens=4, visible_tokens=visible, scale=0.5,
                    queries=[((i * 7) % 13 - 6) / 8 for i in range(heads * 3)],
                    keys=[((i * 11) % 17 - 8) / 4 for i in range(4 * kv_heads * 3)],
                    values=[((i * 13) % 19 - 9) / 2 for i in range(4 * kv_heads * 2)])
        case['output'] = attention(case)
        cases.append(case)
target.with_name('attention.json').write_text(json.dumps(dict(revision=REVISION, cases=cases), indent=2) + '\n')

gdn = function(lib, 'ggml_gated_delta_net', p, [p] * 7 + [c.c_int64])
conv = function(lib, 'ggml_ssm_conv', p, [p, p, p])

def recurrent_graph(kind, case):
    ctx = init(Init(16 * 1024 * 1024, None, False))
    if not ctx:
        raise RuntimeError('ggml context allocation failed')
    try:
        def loaded(shape, values):
            t = tensor3(ctx, 0, *shape)
            raw = (c.c_float * len(values))(*values)
            c.memmove(data(t), raw, c.sizeof(raw))
            return t
        if kind == 'delta':
            d = len(case['query'])
            q = loaded((d, 1, 1), case['query'])
            k = loaded((d, 1, 1), case['key'])
            v = loaded((d, 1, 1), case['value'])
            g = loaded((1, 1, 1), [case['log_decay']])
            beta = loaded((1, 1, 1), [case['beta']])
            s = loaded((d, d, 1), case['state'])
            result = gdn(ctx, q, k, v, g, beta, s, 1)
            count = d + d * d
        else:
            n, width = len(case['input']), case['kernel']
            window = [x for h in range(n) for x in
                      case['history'][h * (width - 1):(h + 1) * (width - 1)] + [case['input'][h]]]
            a = loaded((width, n, 1), window)
            w = loaded((width, n, 1), case['weights'])
            result = conv(ctx, a, w)
            count = n
        g = graph(ctx)
        expand(g, result)
        if compute(ctx, g, 1) != 0:
            raise RuntimeError('CPU recurrent graph failed')
        return list(c.cast(data(result), c.POINTER(c.c_float * count)).contents)
    finally:
        free(ctx)

delta_cases = []
state = [0.5, -1, 2, 0.25]
for decay, beta, value in [(-0.2, 0.3, [1, -2]), (-1, 0.7, [-1, 3]), (0, 1, [2, 1])]:
    case = dict(query=[0.5, -0.25], key=[c.c_float(0.6).value, c.c_float(0.8).value],
                value=value, log_decay=c.c_float(decay).value, beta=c.c_float(beta).value,
                scale=c.c_float(2 ** -0.5).value, state=state)
    packed = recurrent_graph('delta', case)
    case['output'], state = packed[:2], packed[2:]
    case['next_state'] = state
    delta_cases.append(case)
conv_cases = []
history = [1, -2, 3, 4]
for values in [[-1, 2], [0.5, -3], [1, 4]]:
    case = dict(input=values, kernel=3, weights=[1, 2, 4, -0.5, 1, -2], history=history)
    case['output'] = recurrent_graph('convolution', case)
    history = [history[1], values[0], history[3], values[1]]
    case['next_history'] = history
    conv_cases.append(case)
target.with_name('recurrent.json').write_text(json.dumps(dict(revision=REVISION,
    delta=delta_cases, convolution=conv_cases), indent=2) + '\n')
