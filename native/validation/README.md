# Depth Pro native validation

## Canonical model boundary

- Source revision: `9efe5c1def37a26c5367a71df664b18e1306c708`
- Canonical checkpoint: `depth_pro.pt`
- Canonical bytes: 1,904,446,787
- Canonical SHA-256:
  `3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce`
- Inference tensors: 1,119 contiguous FP16 tensors

The native runtime does not parse PyTorch pickle. The development-only
`depth-pro-export-pytorch-v1` converter retains the checkpoint's FP16 tensor
bytes without expanding them to FP32. The hidden derived representation is
keyed by canonical SHA-256, converter ID, and format version, and is associated
with the shared canonical weight rather than exposed as a separate model.

- Derived native bytes: 1,904,197,824
- Archive/container bytes avoided: 248,963

The dependency-free memory-mapped reader validates model and metadata headers,
directory bounds, ranks, dimensions, non-overlap, and exact FP16 payload byte
counts. The catalog checkpoint passes its native model probe.

Full graph CPU-reference validation remains pending.
