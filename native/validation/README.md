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

## Patch-encoder gate

The scalar/multithreaded CPU oracle implements the canonical 384x384
DINOv2-L/16 patch encoder: patch and absolute-position embedding, all 24
attention/MLP blocks, LayerScale, and the four Depth Pro feature taps.

| Block | Relative L1 | Maximum absolute error |
|---:|---:|---:|
| 5 | `2.02407e-6` (`0.000202%`) | `9.77516e-6` |
| 11 | `3.34623e-6` (`0.000335%`) | `1.32322e-5` |
| 17 | `2.81104e-6` (`0.000281%`) | `0.000152588` |
| 23 | `1.45081e-6` (`0.000145%`) | `0.00122070` |

The image pyramid/merge, image and FOV encoders, decoder, and metric conversion
are now implemented through the public dependency-free DLL.

## Full graph gate

The 140x140 catalog canary exercises resize to 1536, 35 overlapping patch
encodings, separate image and FOV encoders, pyramid merge/upsampling, the
five-level convolutional decoder, canonical inverse-depth head, focal
estimation, and metric conversion.

| Metric | Native vs Python CPU |
|---|---:|
| Relative depth L1 | `0.00142429` (`0.142429%`) |
| Maximum absolute depth error | `0.0141956` |
| Focal-length relative error | `1.58642e-6` (`0.000159%`) |
| Correctness-first CPU time | `614.1 s` |

## Vulkan full-graph gate

ABI 2 adds `depth_pro_create_vulkan`. It executes all 37 DINOv2-L encoder
passes, multiscale patch stitching, convolutional decoder, field-of-view head,
and metric conversion on Vulkan. The canonical model remains the shared
checkpoint; the runtime consumes only the bounded, content-addressed `.dpro`
derivation and does not load pickle.

The first Windows hardware canary passes on the Radeon RX 9070:

| Metric | Vulkan vs Python CPU |
|---|---:|
| Relative depth L1 | `0.00142426` (`0.142426%`) |
| Maximum absolute depth error | `0.0141956` |
| Focal-length relative error | `1.98302e-6` (`0.000198%`) |
| Model creation | `2.57 s` |
| Full 140x140 inference | `9.12 s` |

The patch encoder's block 5 and block 11 captures respectively validate at
`2.41602e-6` and `3.33093e-6` relative L1. The Vulkan path currently uses host
input upload and depth readback; external-resource residency is deliberately
not advertised yet.
