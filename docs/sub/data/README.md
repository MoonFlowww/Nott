# Data Loading, Manipulation, and Diagnostics

The `Nott::Data` toolbox wraps dataset ingestion, augmentation, repair, and
analysis helpers so you can prepare tensors before handing them to
[Docs/Training](../training/README.md) or [Docs/Evaluation](../evaluation/README.md).
Factories live under `src/data/details` and are split into four main groups:
`Load`, `Manipulation`, `DimReduction`, and `Check`/`Repair`.

## Dataset Loaders

Every loader is templated on `BufferVRAM`, allowing you to pin the resulting
tensors directly to the device selected by `Core::DevicePolicy`. Leave the flag
at the default (`false`) to receive CPU tensors.

- **`Load::CIFAR10(root, train_fraction, test_fraction, normalise)`**, Parses
  the binary CIFAR-10 batches, optionally downsamples splits, converts images to
  `float32`, and normalises to `[0, 1]` when requested. The helper returns a
  4-tuple `(train_images, train_labels, test_images, test_labels)` ready for
  batching.
- **`Load::MNIST(root, train_fraction, test_fraction, normalise)`**, Reads the
  IDX files distributed by Yann LeCun's site, validates magic numbers and
  shapes, and exposes the digit dataset as `(N, 1, 28, 28)` tensors.
- **`Load::ETTh(csv_path, train_fraction, test_fraction, normalise)`**, Imports
  the Electricity Transformer Temperature (ETT) CSV files, infers feature
  columns, and yields contiguous float tensors for forecasting experiments.
- **`Load::PTBXL(root, low_resolution, train_fraction, normalise, multilabel)`**
 , Handles the PTB-XL ECG archive, resolving `.hea/.dat` pairs, mapping SCP
  statements to the five superclasses, and optionally constructing multilabel
  targets with configurable thresholds.
- **`Load::Universal(root, Descriptor inputs, Descriptor outputs, Global)`** -
  Instantiates descriptor-driven readers for CSV/txt/bin sources, stacks
  tensors, optionally shuffles them, then
  applies the requested `Type::GlobalParameters` split.
  - `Nott::Data::Type` exposes a collection of declarative descriptors so the
      universal loader can reason about heterogeneous sources:
  - `Type::CSV{"dataset.csv", {.columns = {"x0", "x1"}}}`, pick specific
    columns by header (or index) and stream them as `float32` features.
  - `Type::Text{"sentences.txt", {.lowercase = true}}`, treat each line as a
    padded ASCII sequence for NLP-style preprocessing.
  - `Type::Binary{"records.bin", {.type = BinaryDataType::Float32,
    .record_size = 8}}`, reinterpret `.bin` payloads into fixed-size float
    records with optional endianness controls.
  - `Type::PNG{"images/", {.recursive = true}}`, `Type::JPEG{...}`,
    `Type::JPG{...}`, `Type::BMP{...}`, `Type::TIFF{...}`, `Type::PPM{...}`,
    `Type::PGM{...}`, and `Type::PBM{...}`, walk folders of the selected image
    format via OpenCV, convert files to `float32` tensors (optionally grayscale),
    and keep a deterministic ordering. Each descriptor resolves the requested
    directory name anywhere beneath `root`, so callers only need to provide the
    terminal folder (e.g., `"images"` or `"train/images"`) even when the dataset
    adds intermediate layers.
    The loader resolves the requested directory name anywhere beneath `root`, so
    callers only need to provide the terminal folder (e.g., `"images"` or
    `"train/images"`) even when the dataset adds intermediate layers.
  - `Type::GlobalParameters{.train_fraction = 0.8f, .test_fraction = 0.2f}` controls
    the split. Additional files (validation or hold-out) remain untouched if the
    provided fractions do not cover the full dataset.

All loaders guard against missing files and mismatch between data/label counts,
throwing detailed exceptions when directories are incomplete.

## Manipulation and Augmentation

The manipulation module operates directly on tensors and keeps augmentations
deterministic by accepting optional seeds where appropriate.

- **`Flip`**, Mirrors spatial axes based on symbolic tokens (`"x"`, `"y"`,
  `"z"`) or explicit indices. Frequency controls how many batch elements are
  augmented.
- **`Cutout`**, Stochastically masks rectangular patches, either with random
  noise or a constant fill value, and concatenates the augmented samples to the
  original batch.
- **`AtmosphericDrift`**, Injects height-dependent haze gradients with a user
  supplied atmospheric color. `strength`, `drift`, and optional `frequency`
  fields control how pronounced the effect is and how often it is applied to the
  batch.
- **`ChromaticAberration`**, Randomly shifts each channel by up to
  `max_shift_pixels` to simulate lens misalignment. The helper relies on grid
  sampling so tensors keep their original dtype/range.
- **`CLAHE`**, Performs contrast-limited adaptive histogram equalisation over
  a configurable tile grid (`histogram_bins`, `clip_limit`, `tile_grid`). Handy
  for boosting details on X-ray or satellite imagery.
- **`CloudOcclusion`**, Paints soft, perlin-noise-like blobs across the frame
  to mimic satellite clouds. Controls exist for occlusion density, softness, and
  per-sample application frequency.
- **`GridDistortion` / `OpticalDistortion`**, Warp inputs via cubic lattices or
  radial distortion coefficients. Both honour shared random seeds so paired label
  tensors stay aligned.
- **`RandomBrightnessContrast`**, Offsets and scales pixel intensities inside a
  bounded range. Useful for low-light robustness without introducing colour
  casts.
- **`SunAngleJitter`**, Projects synthetic shadows/highlights over outdoor
  datasets by perturbing the virtual sun azimuth/elevation. Tweak `angle_range`
  and `intensity` to match your scene.
- Grid- and optical-distortion helpers accept the same `frequency`/`data_augment`
  toggles as `Cutout`, so you can choose whether they operate on pure training
  batches or also touch evaluation data.
- **`Shuffle` / `Fraction`**, Shuffle pairs of tensors with a shared
  permutation, or extract a fraction of the leading samples while keeping the
  original order intact.
- **`Upsample`**, Thin wrapper around `torch::nn::functional::interpolate` with
  scale-factor semantics.
- **`Grayscale`**, Converts RGB tensors to single-channel luminance using
  configurable weights.


### Formatting & resampling

The `Nott::Data::Transform::Format` namespace surfaces two convenience wrappers
that mirror the new `Layer::Upsample/Downsample` descriptors but can be used on
raw tensors:

- **`Format::Upsample`**, Accepts `ScaleOptions` (`size`, `showprogress`)
  and resizes spatial dimensions while preserving dtype. When `size` is
  omitted the helper simply ensures the tensor is `float32` for downstream
  processing.
- **`Format::Downsample`**, Uses the same options and bilinear interpolation to
  shrink tensors, making it easy to create low-resolution views for multi-scale
  training pipelines.

### Normalisation pipelines

Located under `Data::Normalization`, these routines reshape arbitrary tensor
layouts so the temporal dimension is processed correctly:

- `Zscore` and `EWZscore` compute rolling or exponentially-weighted z-scores.
- `RobustZscore` uses median/MAD estimates for outlier-resistant scaling.
- `Demean` subtracts running or global means without touching variance.
- `StandardizeToTarget` re-scales any series to a desired mean/std pair.
- `FisherTransform` / `InverseFisher` convert bounded oscillators into unbounded
  values (and back) to stabilise financial/cyclical signals before feeding them
  to models.
- `BoxCox`, `YeoJohnson`, and `SignedPower` add power-transform flavours that
  normalise skewed marginals while keeping the code path differentiable.

## Dimensionality Reduction

When prototyping anomaly detection or latent factor models, leverage
`Data::DimReduction`:

- **`RPCA`**, Robust PCA via the inexact augmented Lagrange multiplier method.
  Returns a `(low_rank, sparse)` pair that isolates structured signal from
  transients.
- **`PCA`**, Classical PCA with optional centering and whitening. The helper
  reports principal components, explained variance, singular values, mean, and
  the transformed dataset.

Both routines accept the same `BufferVRAM` template argument as the loaders to
avoid redundant device transfers.

## Data Quality and Repair

- **`Check::Imbalance`**, Produces per-class counts, distributions,
  entropies, and KL divergence between train/test splits.
- **`Check::Shuffled`**, Heuristic to flag datasets that might still be ordered
  by label (useful for verifying randomisation before calling
  [Docs/Training](../training/README.md)).
- **`Check::Size`**, Convenience printer for tensor shapes.

For structured datasets with discrete attributes, `Data::Repair::HoloClean`
implements a lightweight version of the HoloClean probabilistic repair engine:
register functional dependencies, ingest records, and let the factor graph
suggest repairs above a chosen probability threshold.

---

Pair these utilities with [Docs/Links](../links/README.md) to route pre-processed
inputs into complex model branches, or combine them with local scopes from
[Docs/Local](../local/README.md) when different data streams need bespoke training
policies.
