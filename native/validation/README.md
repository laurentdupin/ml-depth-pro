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

The Windows 140x140 hardware canary passes on all three adapters. Large
convolutions and transposed convolutions are divided into bounded spatial
submissions, avoiding both the Windows watchdog and excess retained scratch
memory on the 8 GiB GTX 1080.

| GPU | Relative depth L1 | Maximum absolute | Full inference |
|---|---:|---:|---:|
| Radeon RX 9070 | `0.00142426` (`0.142426%`) | `0.0141956` | `12.23 s` |
| GeForce GTX 1080 | `0.00142439` (`0.142439%`) | `0.0141953` | `60.39 s` |
| Radeon RX 6700 XT | `0.00142394` (`0.142394%`) | `0.0141963` | `18.42 s` |

All three focal-length results have `1.98302e-6` (`0.000198%`) relative
error. Model creation takes approximately 2.1–2.7 seconds.

The patch encoder's block 5 and block 11 captures respectively validate at
`2.41602e-6` and `3.33093e-6` relative L1. The Vulkan path currently uses host
input upload and depth readback; external-resource residency is deliberately
not advertised yet.

## InferBridge BGRA and fixed-FOV contract

ABI 3 adds `depth_pro_infer_bgra8_f32`. It preserves the Python worker's
first-three-byte BGR ordering, performs the same `[0,1]` conversion and
`(x-0.5)/0.5` normalization, and accepts either learned FOV (`0`) or a fixed
FOV in degrees. The default InferBridge setting passes `63`, which skips the
entire learned FOV encoder and uses the caller's focal calibration without
changing the canonical `.pth` to hidden `.dpro` derivation.

A deterministic 32x32 BGRA canary compared fixed-63-degree output with Python
CPU on all three Windows GPUs. Mean metric-depth relative error was `0.498%`.
After the worker's exact `(depth-min)/(25-min)` normalization, maximum output
deviation was `0.01061%` and mean deviation was `0.001962%` on every adapter.
The returned focal length differed from the analytic reference by less than
`0.00001%`. `native/tools/validate_forced_fov.py` reproduces the check.

## Embedded InferBridge harness

The native model DLL exports `ibrh_get_api` for InferBridge harness ABI 1.0.
It accepts the catalog's host-memory BGRA8 image, preserves
`source_frame_id`/timestamp correlation, and returns a leased host-memory FP32
depth image at the source dimensions. `FovEstimation` and `FovForcedValue`
are interpreted with the same defaults as the Python template. After metric
inference, the harness applies the worker's exact
`(depth - depth.min()) / (25 - depth.min())` output transform.

The output lease retains its allocation after the job handle is released.
Capability probing advertises only the implemented synchronous host resource
boundary and one in-flight job. The complete neural graph executes on the
selected Vulkan device, but input upload and output readback remain explicit;
external GPU-resource and cancellation capabilities are not advertised.

The Windows Release build passes `depth_pro_c_abi_smoke`,
`depth_pro_harness_abi_smoke`, and `depth_pro_harness_full_graph` with the
canonical content-addressed `.dpro` derivative. The full-graph harness gate
checks model loading, fixed-FOV parameters, source-size output, correlation,
normalization, finite values, and output-lease lifetime. The direct Python
CPU comparison was also rerun on all three GPUs: maximum worker-output
deviation was `0.000106083` (`0.010609%` of normalized range).
