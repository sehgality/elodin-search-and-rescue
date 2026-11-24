#!/usr/bin/env python3
"""Scan a dumped stablehlo.mlir for stablehlo.concatenate ops whose operand
shapes are inconsistent with valid concatenate semantics (same rank, same
size in every dim except the concat dim) -- this is the kind of mismatch
that would make Elodin's lower_concatenate_nd index out of bounds."""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/elodin_debug/stablehlo.mlir"

dim_re = re.compile(r"dim\s*=\s*(\d+)")
# tensor<AxBxCxf64> or tensor<f64> (rank 0) or tensor<3xi1> etc. -- capture
# the dims part (possibly empty) and the element type separately.
tensor_re = re.compile(r"tensor<([\d x]*)([a-zA-Z0-9_]+)>")


def parse_shape(dims_str: str):
    dims_str = dims_str.strip()
    if not dims_str:
        return []  # rank-0 scalar
    return [int(x) for x in dims_str.split("x") if x != ""]


bad = []
total = 0
parse_errors = 0
with open(path) as f:
    for lineno, line in enumerate(f, 1):
        if "stablehlo.concatenate" not in line:
            continue
        if ":" not in line or "->" not in line:
            continue
        try:
            dim_m = dim_re.search(line)
            if not dim_m:
                continue
            dim = int(dim_m.group(1))

            # Signature is "... : (operand_types) -> result_type"
            sig = line.split(":", 1)[1]
            operands_part, result_part = sig.split("->", 1)

            operand_shapes = [parse_shape(m.group(1)) for m in tensor_re.finditer(operands_part)]
            result_matches = list(tensor_re.finditer(result_part))
            if not result_matches:
                continue
            out_shape = parse_shape(result_matches[0].group(1))
        except Exception as e:
            parse_errors += 1
            continue

        total += 1
        problems = []
        if dim >= len(out_shape):
            problems.append(f"dim={dim} >= out rank {len(out_shape)}")
        for i, shape in enumerate(operand_shapes):
            if len(shape) != len(out_shape):
                problems.append(
                    f"operand {i} rank {len(shape)} != out rank {len(out_shape)} "
                    f"(operand shape {shape}, out shape {out_shape})"
                )
                continue
            for d, (s, o) in enumerate(zip(shape, out_shape)):
                if d == dim:
                    continue
                if s != o:
                    problems.append(
                        f"operand {i} dim {d} size {s} != out dim {d} size {o} "
                        f"(operand shape {shape}, out shape {out_shape}, concat dim={dim})"
                    )
        if dim < len(out_shape):
            concat_sizes = [s[dim] for s in operand_shapes if len(s) > dim]
            if sum(concat_sizes) != out_shape[dim]:
                problems.append(
                    f"concat-dim sizes {concat_sizes} sum to {sum(concat_sizes)} "
                    f"!= out dim {dim} size {out_shape[dim]}"
                )

        if problems:
            bad.append((lineno, line.strip(), problems))

print(f"scanned {total} concatenate ops in {path} ({parse_errors} lines failed to parse)")
print(f"found {len(bad)} with shape inconsistencies\n")
for lineno, line, problems in bad[:20]:
    print(f"--- line {lineno} ---")
    print(line)
    for p in problems:
        print(f"  ! {p}")
    print()

if not bad:
    print("No shape-inconsistent concatenate found by this check.")
    print("The bug may be in how a constant/broadcast operand's VALUE count (not its")
    print("declared TYPE) ends up shorter than its declared shape implies -- which this")
    print("static shape-only check can't see -- or in a different instruction kind entirely.")
