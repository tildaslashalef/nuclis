#!/usr/bin/env python3
# pyright: reportMissingImports=false
# (numpy and gguf-py live in the reference venv and checkout the docstring names.)
"""Write a copy of a GGUF file with every tensor decoded to F32, for an F32 reference pass.

Never imported by the build. Run it with the second llama.cpp checkout's
gguf-py, in the reference venv:

    PYTHONPATH=.reference/llama.cpp-embed/gguf-py .reference/venv-embed/bin/python \\
        scripts/gguf-widen-f32.py IN.gguf OUT.gguf

Q8_0 and BF16 widen exactly (an f16 scale times an int8, the high half of a
single), so the copy holds the same weights. `llama-quantize ... F32` is not
enough: it keeps some tensors (`per_layer_model_proj`) in their own type,
and ggml's CPU matmul then rounds activations to that type.
"""

import argparse

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from gguf.quants import dequantize


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    reader = GGUFReader(args.source)
    writer = GGUFWriter(args.target, arch=reader.fields["general.architecture"].contents())
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        kind = field.types[0]
        if kind == GGUFValueType.ARRAY:
            writer.add_key_value(name, field.contents(), kind, sub_type=field.types[-1])
        else:
            writer.add_key_value(name, field.contents(), kind)
    for tensor in reader.tensors:
        columns = int(tensor.shape[0])
        rows = int(np.prod(tensor.shape[1:]))
        data = dequantize(np.asarray(tensor.data), tensor.tensor_type).astype(np.float32).reshape(rows, columns)
        writer.add_tensor(tensor.name, data, raw_dtype=GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
