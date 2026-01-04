# Nott
Nott is a modern C++ deep-learning framework that layers a strongly typed API over LibTorch. It prioritizes reproducibility, predictable latency, and explicit control over kernels, memory, and optimizer state.
## Why Nott?
* **First-class graph authoring.** Layers and higher-order blocks can be connected as a DAG, letting you express anything from small CNNs to large transformer stacks without wrestling with manual tensor plumbing.
* **Consistent systems model.** Data loaders, augmentations, optimizers, regularizers, and metrics share the same descriptor-driven style so you can mix-and-match building blocks safely.
* **Native performance.** Nott keeps you close to the metal through LibTorch while still providing ergonomic abstractions. Benchmarks at the end of this document detail the runtime overhead compared to pure LibTorch.
* **Not just LibTorch wrappers.** The API covers extensive layer catalogs, attention descriptors, data loaders, and training utilities rather than exposing only a handful of LibTorch layers.
## Quick Start
```cpp
#include <Nott>

int main() {
    Nott::Model model("demo");
    model.use_cuda(torch::cuda::is_available());

    model.add(Nott::Layer::FC({784, 256, true}, Nott::Activation::GeLU));
    model.add(Nott::Layer::Dropout({0.1}));
    model.add(Nott::Layer::FC({256, 10, true}, Nott::Activation::Softmax));

    model.set_optimizer(
        Nott::Optimizer::AdamW({.learning_rate = 1e-3}),
        Nott::LrScheduler::CosineAnnealing({.T_max = 50})
    );

    model.set_loss(Nott::Loss::MSE({}));

    auto [train_images, train_labels, test_images, test_labels] = Nott::Data::Load::MNIST("./datasets", 1.f, 1.f, true);
        
    model.train(train_images, train_labels, {.epoch = 10, .batch_size = 64});

    model.evaluate(test_images, test_labels, Nott::Evaluation::Classification, {
        Nott::Metric::Classification::Precision,
        Nott::Metric::Classification::F1,
        Nott::Metric::Classification::Informedness});
    return 0;
}

```



## Adding Layers and Blocks

The model holds a directed acyclic graph of computational blocks and layers. You can
construct it incrementally with `.add()`.

```cpp
model.add(Nott::Layer::FC({258, 10, /*bias*/true}, Nott::Activation::GeLU, Nott::Initialization::HeNormal))
```
or blocks:
```cpp
model.add(Nott::Block::Sequential({ /*vector field*/
    Nott::Layer::Conv2d({3, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
        Nott::Activation::Identity, Nott::Initialization::HeNormal),
    Nott::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true},
        Nott::Activation::SiLU),
    Nott::Layer::MaxPool2d({{2, 2}, {2, 2}})
}));
```
The framework ships with a rich catalog of layers (see in [Docs/Layers](sub/layer/README.md) or [Docs/Blocks](sub/block/README.md)). It will automatically link linearly every item's called via `.add()`. To rewire the network use `.links()` (see in [Docs/Links](sub/links/README.md)). Multi-head attention descriptors that power the transformer blocks are documented in [Docs/Attention](sub/attention/README.md).


## Configuring Optimization

Optimizer and scheduler choices are set once per model by default. 

The example below pairs
AdamW with cosine annealing warm restarts.

```cpp
model.set_optimizer(
    Nott::Optimizer::AdamW({.learning_rate = 1e-4, .weight_decay = 5e-4}),
    Nott::LrScheduler::CosineAnnealing({
        .T_max = steps_per_epoch * epochs,
        .eta_min = 3e-7,
        .warmup_steps = 5 * steps_per_epoch,
        .warmup_start_factor = 0.1
    })
);
```

Losses and regularization follow the same pattern:

```cpp
model.set_loss(Nott::Loss::CrossEntropy({.label_smoothing = 0.02f}));
model.set_regularization({ /*vector field*/
    Nott::Regularization::SWAG({
        .coefficient = 1e-3,
        .variance_epsilon = 1e-6,
        .start_step = static_cast<size_t>(0.85 * steps_per_epoch * epochs),
        .accumulation_stride = static_cast<size_t>(steps_per_epoch),
        .max_snapshots = 20,
    })
});
```
To see the complete list of Optimizers, Losses or Regularizations and their parameters check [Docs/Optimizer](sub/optimizer/README.md), [Docs/Loss](sub/loss/README.md) and [Docs/Regularization](sub/regularization/README.md)

It is also possible to use multiples configurations over the network, check [Docs/Local](sub/local/README.md)
## Working with Data

The `Nott::Data::Load` namespace includes ready-made loaders for popular datasets
such as MNIST, CIFAR-10, ETTH, PTBXL. Data manipulations (augmentation, shuffling, and splitting) are
exposed through `Nott::Data::Manipulation` utilities, while consistency checks live
under `Nott::Data::Check`.

```cpp
at::Tensor [train_images, train_labels, test_images, test_labels] = Nott::Data::Load::CIFAR10(dataset_root, 1.f, 1.f, true);

at::Tensor [validation_images, validation_labels] = Nott::Data::Manipulation::Fraction(test_images, test_labels, 0.1f);

std::tie(train_images, train_labels) = Nott::Data::Manipulation::Cutout(train_images, train_labels,{-1, -1}, {12, 12}, -1, 1.f, true, false);
```
More information inside [Docs/Data](sub/data/README.md)

## Training

Training is initiated with `model.train`, which accepts tensors and a
`Nott::TrainOptions` struct describing epochs, batch size, graph mode, validation
splits, AMP, and other runtime settings.

```cpp
model.train(train_images, train_labels, {.epoch=120, .batch_size=128, .test={x_val,y_val}});
```
More information in [Docs/Train](sub/training/README.md)
## Evaluation and Metrics

Post-training evaluation is performed with `model.evaluate`, which accepts the test
split, a task type, and a list of metrics. The evaluation API streams batches and
accumulates metrics such as accuracy, precision, recall, calibration errors, and
more.

```cpp
model.evaluate(test_images, test_labels, Nott::Evaluation::Classification, { /*vector field*/
    Nott::Metric::Classification::Precision,
    Nott::Metric::Classification::Recall,
    Nott::Metric::Classification::F1,
    Nott::Metric::Classification::TruePositiveRate,
    Nott::Metric::Classification::LogLoss,
}, {.batch_size = 64});
```
More details inside [Docs/Evaluation](sub/evaluation/README.md)

## Saving and Loading

To keep save your Network use `model.save()` and `model.load()`;
`model.save()` will create a folder of `_Network_Name_` name, and save inside `architecture.json` which correspond to Network layers, dimensions parameters, optimizer used, etc. As well as a `parameter.binary` which store learnable parameters of layers. 

NB: Since `model.load()` read `architecture.json`, you don't need to re-code your network via `model.add()`


```cpp
model.save("PATH");
model.load("PATH"+"/_Network_Name_");
```
`model.save` generates a folder named after the model containing the `architecture.json` (graph, dimensions, optimizer metadata) and `parameters.binary` (learnable weights). Because `model.load` reads the JSON specification, you do not need to recreate the graph via `model.add`. Details live in [Save & Load](sub/saveload/README.md).

## Latency Benchmarks

Results below represent warm runs filtered with a Tukey 0.98 fence on the MNIST workload  
(60k samples, 28×28 | epochs = 100, batch = 64).

Two configurations are reported:

1. **Mixed I/O:** async pinned memory enabled only in **Nott::Train()**.
2. **Unified I/O:** async pinned memory enabled in **all** runners (Nott prebuilt, Nott custom, LibTorch).

---

### 1) Mixed I/O — Async pinned memory only in `Nott::Train()`

| Runner                        | Steps (filtered) | Mean (ms) |    Std |      CV |    P10 |    P50 |    P90 |    P98 |   Mode | Throughput (steps/s) |
|------------------------------|-----------------:|----------:|-------:|--------:|-------:|-------:|-------:|-------:|-------:|----------------------:|
| **Nott — Prebuilt Train()**  |          76 916  | **1.20268** | 0.00157 | **0.00131** | 1.20049 | 1.20302 | 1.20451 | 1.20537 | 1.20398 | **831.47** |
| **Nott — Custom Train()**    |          91 027  | 1.33688 | 0.18792 | 0.14057 | 1.17145 | 1.23006 | 1.65251 | 1.72896 | 1.19031 | 748.01 |
| **LibTorch Raw**             |          90 837  | 1.27572 | 0.18145 | 0.14224 | 1.12161 | 1.16910 | 1.59117 | 1.66006 | 1.13251 | 783.87 |

- **CV** (coefficient of variation) = `Std / Mean`. Lower = less jitter.

#### Overhead (relative to mean latency, positive = slower than reference)

| Comparison                                    |   Value |
|-----------------------------------------------|--------:|
| **Nott** Prebuilt vs **LibTorch** Overhead    |  -5.73% |
| **Nott** Prebuilt vs **Nott** Custom Overhead | -10.04% |
| **Nott** Custom vs **LibTorch** Overhead      |  +4.79% |

In this configuration, Nott’s prebuilt `Train()` benefits from async pinned memory while the other runners do not, so this setup is *favorable* to the prebuilt runner and mainly illustrates the impact of I/O configuration.

---

### 2) Unified I/O — Async pinned memory in all runners

| Runner                        | Steps (filtered) | Mean (ms) |     Std |      CV |    P10 |    P50 |    P90 |    P98 |   Mode | Throughput (steps/s) |
|------------------------------|-----------------:|----------:|--------:|--------:|-------:|-------:|-------:|-------:|-------:|----------------------:|
| **Nott — Prebuilt Train()**  |          71 288  | 1.06486 | 0.00184 | 0.00172 | 1.06275 | 1.06475 | 1.06702 | 1.06889 | 1.06298 | 939.09 |
| **Nott — Custom Train()**    |          75 622  | 1.06443 | 0.01764 | 0.01657 | 1.04117 | 1.06435 | 1.08850 | 1.10319 | 1.06208 | 939.47 |
| **LibTorch Raw**             |          80 820  | **1.02841** | 0.00512 | 0.00498 | 1.02150 | 1.02813 | 1.03556 | 1.03934 | 1.02704 | **972.37** |

- **CV** (coefficient of variation) = `Std / Mean`. Lower = less jitter.

#### Overhead (relative to mean latency, positive = slower than reference)

| Comparison                                    |   Value |
|-----------------------------------------------|--------:|
| **Nott** Prebuilt vs **LibTorch** Overhead    |  +3.54% |
| **Nott** Prebuilt vs **Nott** Custom Overhead |  +0.04% |
| **Nott** Custom vs **LibTorch** Overhead      |  +3.50% |

With identical pinned-memory settings, Nott’s prebuilt `Train()` stays within a few percent of raw LibTorch in mean latency and throughput, while keeping jitter extremely low.

---

> **Note on variability.** These numbers are **relative**, not absolute. Modern hardware is noisy: clocks, power limits, thermals and OS scheduling all drift, so repeated runs with identical settings produce slightly different latency distributions. The robust takeaway across both tables is that Nott’s wrapper adds at most a few percent overhead compared to raw LibTorch, and can even be faster under certain I/O configurations, while preserving very low jitter.

Source: [`test/speedtest.cpp`](../test/speedtest.cpp)



## Research References

Nott’s backend modules follow the algorithms as described in the modern deep-learning literature. For each mechanism we cite a **canonical paper or textbook** (not always the first historical appearance) and link both to the reference and to the implementation file.

### Activations
* **Gaussian Error Linear Unit (GeLU).** Dan Hendrycks, Kevin Gimpel. “Gaussian Error Linear Units (GELUs).” [_arXiv:1606.08415_](https://arxiv.org/abs/1606.08415). (Module: [_Nott::Activation::GeLU_](../src/activation/details/gelu.hpp)).
* **Gated Linear Unit (GLU).** Yann N. Dauphin et al. “Language Modeling with Gated Convolutional Networks.” [_arXiv:1612.08083_](https://arxiv.org/abs/1612.08083). (Module: [_Nott::Activation::GLU_](../src/activation/details/glu.hpp)).
* **Mish.** Diganta Misra. “Mish: A Self Regularized Non-Monotonic Neural Activation Function.” [_arXiv:1908.08681_](https://arxiv.org/abs/1908.08681). (Module: [_Nott::Activation::Mish_](../src/activation/details/mish.hpp)).
* **SiLU / Swish.** Prajit Ramachandran, Barret Zoph, Quoc V. Le. “Searching for Activation Functions.” [_arXiv:1710.05941_](https://arxiv.org/abs/1710.05941). (Modules: [_Nott::Activation::SiLU_](../src/activation/details/silu.hpp), [_Nott::Activation::Swish_](../src/activation/details/swish.hpp)).
* **SwiGLU.** Aakanksha Chowdhery et al. “PaLM: Scaling Language Modeling with Pathways.” [_arXiv:2204.02311_](https://arxiv.org/abs/2204.02311). (Module: [_Nott::Activation::SwiGLU_](../src/activation/details/swiglu.hpp)).

### Transformers
* **Classic Transformer.** Ashish Vaswani et al. “Attention Is All You Need.” [_arXiv:1706.03762_](https://arxiv.org/abs/1706.03762). (Module: [_Nott::Block::Transformer::Classic_](../src/block/details/transformers/classic.hpp)).
* **BERT.** Jacob Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” [_arXiv:1810.04805_](https://arxiv.org/abs/1810.04805). (Module: [_Nott::Block::Transformer::BERT_](../src/block/details/transformers/bert.hpp)).
* **Transformer++.** Hanxiao Liu et al. “Transformer++: Improving Parallelism, Efficiency and Performance of Transformer Models.” [_arXiv:2003.04974_](https://arxiv.org/abs/2003.04974). (Module: [_Nott::Block::Transformer::PlusPlus_](../src/block/details/transformers/plusplus.hpp)).
* **Longformer-XL.** Iz Beltagy et al. “Longformer: The Long-Document Transformer.” [_arXiv:2004.05150_](https://arxiv.org/abs/2004.05150) and Zihang Dai et al. “Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.” [_arXiv:1901.02860_](https://arxiv.org/abs/1901.02860). (Module: [_Nott::Block::Transformer::LongformerXL_](../src/block/details/transformers/longformer_xl.hpp)).
* **Vision Transformer.** Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” [_arXiv:2010.11929_](https://arxiv.org/abs/2010.11929). (Module: [_Nott::Block::Transformer::Vision_](../src/block/details/transformers/vision.hpp)).
* **Perceiver.** Andrew Jaegle et al. “Perceiver: General Perception with Iterative Attention.” [_arXiv:2103.03206_](https://arxiv.org/abs/2103.03206). (Module: [_Nott::Block::Transformer::Perceiver_](../src/block/details/transformers/perceiver.hpp)).
* **Mamba.** Albert Gu et al. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” [_arXiv:2312.00752_](https://arxiv.org/abs/2312.00752). (Module: [_Nott::Block::Transformer::Mamba_](../src/block/details/transformers/mamba.hpp)).
* **Energy-Based Transformer (EBT).** Mikael Haziza et al. “Energy-Based Transformers.” [_arXiv:2507.02092_](https://arxiv.org/abs/2507.02092). (Module: [_Nott::Block::Transformer::EBT_](../src/block/details/transformers/ebt.hpp)).
* **Atlas.** Theodore Sumers et al. “Atlas: Learning to Optimally Memorize the Context at Test Time.” [_arXiv:2505.23735_](https://arxiv.org/abs/2505.23735). (Module: [_Nott::Block::Transformer::Atlas_](../src/block/details/transformers/atlas.hpp)).
* **Titan.** Zhifan Liu et al. “Titan: Scaling Language Model Training with Real-Time, Low-Latency Adaptation.” [_arXiv:2501.00663_](https://arxiv.org/abs/2501.00663). (Module: [_Nott::Block::Transformer::Titan_](../src/block/details/transformers/titan.hpp)).

### Layers
* **Dropout.** Nitish Srivastava et al. “Dropout: A Simple Way to Prevent Neural Networks from Overfitting.” [_arXiv:1207.0580_](https://arxiv.org/abs/1207.0580). (Module: [_Nott::Layer::Dropout_](../src/layer/details/dropout.hpp)).
* **Batch Normalization.** Sergey Ioffe, Christian Szegedy. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” [_arXiv:1502.03167_](https://arxiv.org/abs/1502.03167). (Module: [_Nott::Layer::BatchNorm_](../src/layer/details/batchnorm.hpp)).
* **Instance Normalization.** Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky. “Instance Normalization: The Missing Ingredient for Fast Stylization.” [_arXiv:1607.08022_](https://arxiv.org/abs/1607.08022). (Module: [_Nott::Layer::InstanceNorm_](../src/layer/details/instancenorm.hpp)).
* **Convolutional Layers.** Kunihiko Fukushima. “Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.” [_Biological Cybernetics 1980_](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf) (early convolution + pooling-like architecture); Yann LeCun et al. “Gradient-Based Learning Applied to Document Recognition.” [_Proceedings of the IEEE 1998_](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) (modern gradient-trained CNNs with conv + pooling + FC); Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. “ImageNet Classification with Deep Convolutional Neural Networks.” [_NeurIPS 2012_](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (large-scale deep CNN on GPUs). (Modules: [_Nott::Layer::Conv2d_](../src/layer/details/conv.hpp) and variants).
* **Fully Connected / Perceptron Layers.** Frank Rosenblatt. “The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.” [_Rosenblatt 1958_](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf) (single-layer perceptron); David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams. “Learning Representations by Back-Propagating Errors.” _Nature_ 1986 (multilayer perceptrons with backpropagation). (Module: [_Nott::Layer::FC_](../src/layer/details/fc.hpp)).
* **Pooling Layers.** Kunihiko Fukushima. “Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.” 1980 (early “subsampling” / pooling); Yann LeCun et al. “Gradient-Based Learning Applied to Document Recognition.” [_Proceedings of the IEEE 1998_](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) (max/average pooling in modern CNNs). (Modules: [_Nott::Layer::MaxPool2d_](../src/layer/details/pooling.hpp), [_Nott::Layer::AvgPool2d_](../src/layer/details/pooling.hpp)).
* **Recurrent Layers.** Jeffrey L. Elman. “Finding Structure in Time.” _Cognitive Science_ 1990 (Elman RNN); Sepp Hochreiter, Jürgen Schmidhuber. “Long Short-Term Memory.” [_Neural Computation 1997_](https://www.bioinf.jku.at/publications/older/2604.pdf); Kyunghyun Cho et al. “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.” [_arXiv:1406.1078_](https://arxiv.org/abs/1406.1078). (Modules: [_Nott::Layer::RNN_](../src/layer/details/recurrent.hpp), [_Nott::Layer::LSTM_](../src/layer/details/recurrent.hpp), [_Nott::Layer::GRU_](../src/layer/details/recurrent.hpp)).
* **Positional Encoding.** Ashish Vaswani et al. “Attention Is All You Need.” [_arXiv:1706.03762_](https://arxiv.org/abs/1706.03762). (Module: [_Nott::Layer::PositionalEncoding_](../src/layer/details/positional_encoding.hpp)).
* **Structured State Spaces (S4).** Albert Gu et al. “Efficiently Modeling Long Sequences with Structured State Spaces.” [_arXiv:2111.00396_](https://arxiv.org/abs/2111.00396). (Module: [_Nott::Layer::S4_](../src/layer/details/s4.hpp)).
* **Patch (Un)Embedding.** Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” [_arXiv:2010.11929_](https://arxiv.org/abs/2010.11929). (Modules: [_Nott::Layer::PatchUnembed_](../src/layer/details/patch_unembed.hpp), [_Nott::Layer::Resizing_](../src/layer/details/resizing.hpp)).
* **Flatten and Reduce.** Generic tensor reshaping and spatial reduction operations used throughout early CNN architectures (e.g., LeCun et al. 1998). (Modules: [_Nott::Layer::Flatten_](../src/layer/details/flatten.hpp), [_Nott::Layer::Reduce_](../src/layer/details/reduce.hpp)).

### Losses

For classical statistical losses (CE, MSE, MAE, logistic), the references below are **standard ML expositions**, not the original 19th–20th century statistical papers.

* **Cross-Entropy / Negative Log Likelihood.** Rooted in information theory and relative entropy: Claude E. Shannon. “A Mathematical Theory of Communication.” [_Bell System Technical Journal 1948_](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf); Solomon Kullback, Richard A. Leibler. “On Information and Sufficiency.” [_Annals of Mathematical Statistics 1951_](https://scispace.com/pdf/on-information-and-sufficiency-3o6xv2h9bw.pdf). For a modern ML treatment we follow: David J. C. MacKay. “Information Theory, Inference, and Learning Algorithms.” [_2003 Text_](https://www.inference.org.uk/itprnn/book.pdf). (Modules: [_Nott::Loss::CE_](../src/loss/details/ce.hpp), [_Nott::Loss::NLL_](../src/loss/details/nll.hpp)).
* **Binary Cross-Entropy / Logistic Loss.** Originating from logistic models and Bernoulli log-likelihood: Joseph Berkson. “Application of the Logistic Function to Bio-Assay.” [_Journal of the American Statistical Association 1944_](https://www.tandfonline.com/doi/abs/10.1080/01621459.1944.10500699) (introduces and justifies the logistic / logit model for bio-assay). For a standard applied treatment we follow: David W. Hosmer, Stanley Lemeshow. “Applied Logistic Regression.” [_2000 Text_](https://onlinelibrary.wiley.com/doi/book/10.1002/0471722146). (Module: [_Nott::Loss::BCE_](../src/loss/details/bce.hpp)).
* **Categorical Cross-Entropy.** Multinomial / categorical negative log-likelihood in the sense of classical likelihood theory: Ronald A. Fisher. “On the Mathematical Foundations of Theoretical Statistics.” [_Phil. Trans. of the Royal Society A 1922_](http://www.stats.org.uk/statistical-inference/Fisher1922.pdf). For the modern softmax cross-entropy formulation in ML we follow: Christopher M. Bishop. “Pattern Recognition and Machine Learning.” [_2006 Text_](https://link.springer.com/book/10.1007/978-0-387-45528-0). (Module: [_Nott::Loss::CCE_](../src/loss/details/cce.hpp)).

* **Mean Squared Error / Mean Absolute Error.** Rooted in classical least-squares and least-absolute-deviations: Adrien-Marie Legendre. “Nouvelles méthodes pour la détermination des orbites des comètes.” [_1805_](https://books.google.fr/books?id=Ia8WAAAAQAAJ&printsec=frontcover&hl=fr#v=onepage&q&f=false); Carl Friedrich Gauss. “Theoria motus corporum coelestium in sectionibus conicis solem ambientium.” [_1809_](https://archive.org/details/bub_gb_ORUOAAAAQAAJ/mode/2up) (formalizing least squares / squared-error minimization). For a modern statistical learning treatment we follow: Vladimir Vapnik. “The Nature of Statistical Learning Theory.” [_1995 Text_](https://link.springer.com/book/10.1007/978-1-4757-2440-0). (Modules: [_Nott::Loss::MSE_](../src/loss/details/mse.hpp), [_Nott::Loss::MAE_](../src/loss/details/mae.hpp)).
* **Smooth L1 (Huber) Loss.** Peter J. Huber. “Robust Estimation of a Location Parameter.” [_Annals of Mathematical Statistics 1964_](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/robust-estimation-of-a-location-parameter/10.1214/aoms/1177703732.full) (original Huber loss); Ross Girshick. “Fast R-CNN.” [_arXiv:1504.08083_](https://arxiv.org/abs/1504.08083) (popular Smooth L1 implementation for bounding-box regression). (Module: [_Nott::Loss::SmoothL1_](../src/loss/details/smooth_l1.hpp)).
* **Dice Loss.** Fausto Milletari et al. “V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.” [_arXiv:1606.04797_](https://arxiv.org/abs/1606.04797). (Module: [_Nott::Loss::Dice_](../src/loss/details/dice.hpp)).
* **Tversky Loss.** Seyed Sadegh Mohseni Salehi et al. “Tversky loss function for image segmentation using 3D fully convolutional deep networks.” [_arXiv:1706.05721_](https://arxiv.org/abs/1706.05721). (Based on the Tversky index from Amos Tversky, “Features of Similarity,” _Psychological Review_ 1977.) (Module: [_Nott::Loss::Tversky_](../src/loss/details/tversky.hpp)).
* **Lovász-Softmax.** Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko. “The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks.” [_arXiv:1705.08790_](https://arxiv.org/abs/1705.08790). (Module: [_Nott::Loss::LovaszSoftmax_](../src/loss/details/lovasz_softmax.hpp)).
* **Cosine Embedding Loss.** Raia Hadsell, Sumit Chopra, Yann LeCun. “Dimensionality Reduction by Learning an Invariant Mapping.” [_CVPR 2006_](https://ieeexplore.ieee.org/document/1640964). (Module: [_Nott::Loss::CosineEmbedding_](../src/loss/details/cosine_embedding.hpp)).
* **Margin Ranking Loss.** Thorsten Joachims. “Optimizing Search Engines Using Clickthrough Data.” [_KDD 2002_](https://dl.acm.org/doi/10.1145/775047.775067) (introducing large-margin ranking). (Module: [_Nott::Loss::MarginRanking_](../src/loss/details/margin_ranking.hpp)).
* **Kullback–Leibler Divergence.** Solomon Kullback, Richard A. Leibler. “On Information and Sufficiency.” [_Annals of Mathematical Statistics 1951_](https://scispace.com/pdf/on-information-and-sufficiency-3o6xv2h9bw.pdf). (Module: [_Nott::Loss::KL_](../src/loss/details/kl.hpp)).

### Learning Rate Schedulers
* **Cosine Annealing with Warm Restarts.** Ilya Loshchilov, Frank Hutter. “SGDR: Stochastic Gradient Descent with Warm Restarts.” [_arXiv:1608.03983_](https://arxiv.org/abs/1608.03983). (Module: [_Nott::LrScheduler::CosineAnnealing_](../src/lrscheduler/details/cosineannealing.hpp)).
* **Exponential Decay.** Yann LeCun, Léon Bottou, Genevieve B. Orr, Klaus-Robert Müller. “Efficient BackProp.” [_Tricks of the Trade 2012_](https://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf). (Module: [_Nott::LrScheduler::Exponential_](../src/lrscheduler/details/exponential.hpp)).

### Optimizers
* **Adafactor.** Noam Shazeer, Mitchell Stern. “Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.” [_arXiv:1804.04235_](https://arxiv.org/abs/1804.04235). (Module: [_Nott::Optimizer::Adafactor_](../src/optimizer/details/adafactor.hpp)).
* **LAMB.** Yang You et al. “Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.” [_arXiv:1904.00962_](https://arxiv.org/abs/1904.00962). (Module: [_Nott::Optimizer::LAMB_](../src/optimizer/details/lamb.hpp)).
* **Lion.** Qianqian Gu et al. “Symbolic Discovery of Optimization Algorithms.” [_arXiv:2302.06675_](https://arxiv.org/abs/2302.06675). (Module: [_Nott::Optimizer::Lion_](../src/optimizer/details/lion.hpp)).
* **Adam.** Diederik P. Kingma, Jimmy Ba. “Adam: A Method for Stochastic Optimization.” [_arXiv:1412.6980_](https://arxiv.org/abs/1412.6980). (Module: [_Nott::Optimizer::Adam_](../src/optimizer/details/adam.hpp)).
* **AdamW.** Ilya Loshchilov, Frank Hutter. “Decoupled Weight Decay Regularization.” [_arXiv:1711.05101_](https://arxiv.org/abs/1711.05101). (Module: [_Nott::Optimizer::AdamW_](../src/optimizer/details/adam.hpp)).
* **AdaGrad.** John Duchi, Elad Hazan, Yoram Singer. “Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.” [_JMLR 2011_](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf). (Module: [_Nott::Optimizer::AdaGrad_](../src/optimizer/details/adagrad.hpp)).
* **RMSProp.** Geoffrey Hinton. “Neural Networks for Machine Learning” (Coursera lecture 6.5), 2012. [_Lecture Notes_](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf). (Module: [_Nott::Optimizer::RMSprop_](../src/optimizer/details/rmsprop.hpp)).
* **Stochastic Gradient Descent with Momentum.** Boris T. Polyak. “Some Methods of Speeding up the Convergence of Iteration Methods.” [_USSR Comput. Math. and Math. Phys. 1964_](https://papers.baulab.info/papers/also/Polyak-1964.pdf). (Module: [_Nott::Optimizer::SGD_](../src/optimizer/details/sgd.hpp)).
* **Muon, AdaMuon, MuonManifold.** Pavol Bielik et al. “Muon: Muon Momentum + Adaptive Manifold Optimization.” [_arXiv:2502.16982_](https://arxiv.org/abs/2502.16982); “AdaMuon: Adaptive Momentum for Muon Optimizer.” [_arXiv:2507.11005_](https://arxiv.org/abs/2507.11005); Thinking Machines “Modular Manifolds.” [_Modular Manifolds_](https://www.thinkingmachines.ai/blog/modular-manifolds/). (Module: [_Nott::Optimizer::Muon_](../src/optimizer/details/muon.hpp)).
* **Sophia.** Shuchen Zhang et al. “Sophia: A Scalable Stochastic Second-Order Optimizer for Language Models.” [_arXiv:2305.14342_](https://arxiv.org/abs/2305.14342). (Module: [_Nott::Optimizer::Sophia_](../src/optimizer/details/sophia.hpp)).

### Regularization
* **Spectral Normalization.** Takeru Miyato et al. “Spectral Normalization for Generative Adversarial Networks.” [_arXiv:1802.05957_](https://arxiv.org/abs/1802.05957). (Module: [_Nott::Regularization::SpectralNorm_](../src/regularization/details/spectralnorm.hpp)).
* **Stochastic Weight Averaging (SWA).** Pavel Izmailov et al. “Averaging Weights Leads to Wider Optima in Deep Learning.” [_arXiv:1803.05407_](https://arxiv.org/abs/1803.05407). (Module: [_Nott::Regularization::SWA_](../src/regularization/details/swa.hpp)).
* **SWAG.** Wesley J. Maddox et al. “SWAG: A Simple Baseline for Bayesian Uncertainty in Deep Learning.” [_arXiv:1902.02476_](https://arxiv.org/abs/1902.02476). (Module: [_Nott::Regularization::SWAG_](../src/regularization/details/swag.hpp)).
* **TRADES.** Hongyang Zhang et al. “Theoretically Principled Trade-off between Robustness and Accuracy.” [_arXiv:1901.08573_](https://arxiv.org/abs/1901.08573). (Module: [_Nott::Regularization::TRADES_](../src/regularization/details/trades.hpp)).
* **Virtual Adversarial Training (VAT).** Takeru Miyato et al. “Virtual Adversarial Training: A Regularization Method for Supervised and Semi-supervised Learning.” [_arXiv:1704.03976_](https://arxiv.org/abs/1704.03976). (Module: [_Nott::Regularization::VAT_](../src/regularization/details/vat.hpp)).
* **Elastic Net.** Hui Zou, Trevor Hastie. “Regularization and Variable Selection via the Elastic Net.” [_JRSS B 2005_](https://miguelbiron.github.io/docs/STAT548_report_2_Elasticnet.pdf). (Module: [_Nott::Regularization::ElasticNet_](../src/regularization/details/elasticnet.hpp)).
* **L1 / L2 Penalties.** Andrew Y. Ng. “Feature selection, L1 vs. L2 regularization, and rotational invariance.” [_ICML 2004_](https://icml.cc/Conferences/2004/proceedings/papers/354.pdf). (Modules: [_Nott::Regularization::L1_](../src/regularization/details/l1.hpp), [_Nott::Regularization::L2_](../src/regularization/details/l2.hpp)).
* **Group Lasso.** Ming Yuan, Yi Lin. “Model selection and estimation in regression with grouped variables.” [_JRSS B 2006_](http://6577418.s21d-6.faiusrd.com/61/ABUIABA9GAAgov6wvgUo57qaYQ.pdf). (Module: [_Nott::Regularization::GroupLasso_](../src/regularization/details/grouplasso.hpp)).
**** **Max-Norm Constraints.** George E. Dahl et al. “Improving Deep Neural Networks for LVCSR using Maxout and Dropout.” [_ICASSP 2013_](https://ieeexplore.ieee.org/document/6639346) (popularized max-norm constraints). (Module: [_Nott::Regularization::MaxNorm_](../src/regularization/details/maxnorm.hpp)).
* **Orthogonality Regularization.** Ankit Bansal, Daniel Chen, David Jacobs. “Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?” [_NeurIPS 2018_](https://arxiv.org/pdf/1810.09102). (Module: [_Nott::Regularization::Orthogonality_](../src/regularization/details/orthogonality.hpp)).
* **Nuclear Norm.** Nathan Srebro, Jason Rennie, Tommi Jaakkola. “Maximum-Margin Matrix Factorization.” [_NeurIPS 2004_](http://qwone.com/~jason/papers/nips04-mmmf.pdf). (Module: [_Nott::Regularization::NuclearNorm_](../src/regularization/details/nuclearnorm.hpp)).
* **Jacobian Regularization.** Patrice Simard et al. “Best Practices for Convolutional Neural Networks applied to Visual Document Analysis.” [_ICDAR 2003_](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Simard.pdf). (Module: [_Nott::Regularization::Jacobian_](../src/regularization/details/jacobian.hpp)).
* **Decorrelation (DeCov).** Yingying C. Sun, Andrew L. Maas, Surya Ganguli, Andrew Y. Ng. “DeCov: A Simple Way to Improve Generalization.” [_arXiv:1511.06068_](https://arxiv.org/abs/1511.06068). (Module: [_Nott::Regularization::Decov_](../src/regularization/details/decov.hpp)).
* **Fisher and Sharpness-aware FGE.** Timur Garipov et al. “Loss Surfaces, Mode Connectivity, and Fast Geometric Ensembling.” [_arXiv:1802.10026_](https://arxiv.org/abs/1802.10026); Christian Liebel, Eva Müller. “Sharpness-Aware Training for Fast Geometric Ensembling.” [_arXiv:2303.00595_](https://arxiv.org/abs/2303.00595). (Modules: [_Nott::Regularization::FGE_](../src/regularization/details/fge.hpp), [_Nott::Regularization::SFGE_](../src/regularization/details/sfge.hpp)).
* **Elastic Weight Consolidation.** James Kirkpatrick et al. “Overcoming catastrophic forgetting in neural networks.” [_PNAS 2017_](https://www.pnas.org/doi/10.1073/pnas.1611835114). (Module: [_Nott::Regularization::EWC_](../src/regularization/details/ewc.hpp)).
* **Memory Aware Synapses.** Rahaf Aljundi et al. “Memory Aware Synapses: Learning what (not) to forget.” [_ECCV 2018_](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf). (Module: [_Nott::Regularization::MAS_](../src/regularization/details/mas.hpp)).
* **Synaptic Intelligence.** Friedemann Zenke, Ben Poole, Surya Ganguli. “Continual Learning Through Synaptic Intelligence.” [_arXiv:1703.04200_](https://arxiv.org/abs/1703.04200). (Module: [_Nott::Regularization::SI_](../src/regularization/details/si.hpp)).
* **L0 Hard Concrete Gates.** Christos Louizos, Max Welling, Diederik P. Kingma. “Learning Sparse Neural Networks through L0 Regularization.” [_arXiv:1712.01312_](https://arxiv.org/abs/1712.01312). (Module: [_Nott::Regularization::L0HardConcrete_](../src/regularization/details/l0hardconcrete.hpp)).
* **Kullback–Leibler Sparsity.** Andrew Ng. “Sparse Autoencoder.” [_CS294A Lecture Notes 2011_](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf). (Module: [_Nott::Regularization::KLSparsity_](../src/regularization/details/klsparsity.hpp)).
* **R1 / R2 Gradient Penalties.** Lars Mescheder, Sebastian Nowozin, Andreas Geiger. “Which Training Methods for GANs do actually Converge?” [_arXiv:1801.04406_](https://arxiv.org/abs/1801.04406). (Modules: [_Nott::Regularization::R1_](../src/regularization/details/r1.hpp), [_Nott::Regularization::R2_](../src/regularization/details/r2.hpp)).
* **WGAN-GP.** Ishaan Gulrajani et al. “Improved Training of Wasserstein GANs.” [_arXiv:1704.00028_](https://arxiv.org/abs/1704.00028). (Module: [_Nott::Regularization::WGANGP_](../src/regularization/details/wgangp.hpp)).
