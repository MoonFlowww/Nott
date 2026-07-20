cmake_minimum_required(VERSION 3.18)
project(Nott CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

add_compile_options($<$<CONFIG:Release>:-O3>)

# ---- Dependencies --------------------------------------------------------
# Adjust this cache variable (or pass -DTorch_DIR=... / extend
# CMAKE_PREFIX_PATH) if LibTorch is not installed at the default location.
set(_libtorch_root "/opt/libtorch" CACHE PATH "Fallback LibTorch install root, used when Torch_DIR is not already set")
if(EXISTS "${_libtorch_root}/share/cmake/Torch/TorchConfig.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${_libtorch_root}")
    set(Torch_DIR "${_libtorch_root}/share/cmake/Torch")
endif()

find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

# ---- Nott library ----------------------------------------------------------
# Nott is a header-only framework: src/Nott.cpp is just a translation-unit
# anchor (no real code lives there). Every header below is attached to the
# target purely so IDEs (CLion, etc.) index and display them under Nott --
# none of them are compiled independently; they are pulled in by whichever
# example #includes them. See docs/sub/build/README.md for the full build
# guide.
add_library(Nott SHARED
    src/Nott.cpp
    src/activation/activation.hpp
    src/activation/apply.hpp
    src/activation/details/dsilu.hpp
    src/activation/details/gelu.hpp
    src/activation/details/glu.hpp
    src/activation/details/leaky_relu.hpp
    src/activation/details/mish.hpp
    src/activation/details/psilu.hpp
    src/activation/details/relu.hpp
    src/activation/details/sigmoid.hpp
    src/activation/details/silu.hpp
    src/activation/details/softmax.hpp
    src/activation/details/swiglu.hpp
    src/activation/details/swish.hpp
    src/activation/details/tanh.hpp
    src/attention/attention.hpp
    src/attention/builder.hpp
    src/attention/details/head.hpp
    src/attention/details/kernel.hpp
    src/attention/details/latent.hpp
    src/block/block.hpp
    src/block/details/blocks/residual.hpp
    src/block/details/blocks/sequential.hpp
    src/block/details/transformers/atlas.hpp
    src/block/details/transformers/bert.hpp
    src/block/details/transformers/classic.hpp
    src/block/details/transformers/ebt.hpp
    src/block/details/transformers/longformer_xl.hpp
    src/block/details/transformers/mamba.hpp
    src/block/details/transformers/perceiver.hpp
    src/block/details/transformers/plusplus.hpp
    src/block/details/transformers/titan.hpp
    src/block/details/transformers/vision.hpp
    src/calibration/calibration.hpp
    src/calibration/details/temperature_scaling.hpp
    src/common/graph.hpp
    src/common/local.hpp
    src/common/save_load.hpp
    src/common/streaming.hpp
    src/common/stresstest.hpp
    src/core.hpp
    src/data/data.hpp
    src/data/details/generation.hpp
    src/data/details/repair.hpp
    src/data/load/load.hpp
    src/data/load/types.hpp
    src/data/transform/augment/atmospheric_drift.hpp
    src/data/transform/augment/chromatic_aberration.hpp
    src/data/transform/augment/clahe.hpp
    src/data/transform/augment/cloud_occlusion.hpp
    src/data/transform/augment/common.hpp
    src/data/transform/augment/cutout.hpp
    src/data/transform/augment/flip.hpp
    src/data/transform/augment/grid_distortion.hpp
    src/data/transform/augment/optical_distortion.hpp
    src/data/transform/augment/random_brightness_contrast.hpp
    src/data/transform/augment/registry.hpp
    src/data/transform/augment/sun_angle_jitter.hpp
    src/data/transform/dimensionality_reduction.hpp
    src/data/transform/dimensionality_reduction/common.hpp
    src/data/transform/dimensionality_reduction/pca.hpp
    src/data/transform/dimensionality_reduction/registry.hpp
    src/data/transform/dimensionality_reduction/rpca.hpp
    src/data/transform/format/format.hpp
    src/data/transform/manipulation.hpp
    src/data/transform/normalization/common.hpp
    src/data/transform/normalization/ehlers.hpp
    src/data/transform/normalization/power.hpp
    src/data/transform/normalization/registry.hpp
    src/data/transform/normalization/zscore.hpp
    src/evaluation/details/classification.hpp
    src/evaluation/details/timeserie.hpp
    src/evaluation/evaluation.hpp
    src/initialization/apply.hpp
    src/initialization/initialization.hpp
    src/layer/details/batchnorm.hpp
    src/layer/details/conv.hpp
    src/layer/details/dropout.hpp
    src/layer/details/fc.hpp
    src/layer/details/flatten.hpp
    src/layer/details/instancenorm.hpp
    src/layer/details/patch_unembed.hpp
    src/layer/details/pooling.hpp
    src/layer/details/positional_encoding.hpp
    src/layer/details/recurrent.hpp
    src/layer/details/reduce.hpp
    src/layer/details/resizing.hpp
    src/layer/details/s4.hpp
    src/layer/layer.hpp
    src/layer/registry.hpp
    src/loss/details/bce.hpp
    src/loss/details/cce.hpp
    src/loss/details/ce.hpp
    src/loss/details/cosine_embedding.hpp
    src/loss/details/dice.hpp
    src/loss/details/helper.hpp
    src/loss/details/kl.hpp
    src/loss/details/lovasz_softmax.hpp
    src/loss/details/mae.hpp
    src/loss/details/margin_ranking.hpp
    src/loss/details/mse.hpp
    src/loss/details/nll.hpp
    src/loss/details/reduction.hpp
    src/loss/details/smooth_l1.hpp
    src/loss/details/tversky.hpp
    src/loss/loss.hpp
    src/lrscheduler/details/common.hpp
    src/lrscheduler/details/cosineannealing.hpp
    src/lrscheduler/details/exponential.hpp
    src/lrscheduler/lrscheduler.hpp
    src/lrscheduler/registry.hpp
    src/metric/metric.hpp
    src/optimizer/details/adafactor.hpp
    src/optimizer/details/adagrad.hpp
    src/optimizer/details/adam.hpp
    src/optimizer/details/lamb.hpp
    src/optimizer/details/lion.hpp
    src/optimizer/details/muon.hpp
    src/optimizer/details/rmsprop.hpp
    src/optimizer/details/sgd.hpp
    src/optimizer/details/sophia.hpp
    src/optimizer/optimizer.hpp
    src/optimizer/registry.hpp
    src/plot/details/data.hpp
    src/plot/details/reliability.hpp
    src/plot/details/reliability/reliability_curve_utils.hpp
    src/plot/details/reliability/reliability_det.hpp
    src/plot/details/reliability/reliability_gradcam.hpp
    src/plot/details/reliability/reliability_lime.hpp
    src/plot/details/reliability/reliability_pr.hpp
    src/plot/details/reliability/reliability_roc.hpp
    src/plot/details/reliability/reliability_youdens.hpp
    src/plot/details/statistics.hpp
    src/plot/details/training.hpp
    src/plot/plot.hpp
    src/regularization/apply.hpp
    src/regularization/details/centeringvariance.hpp
    src/regularization/details/common.hpp
    src/regularization/details/decov.hpp
    src/regularization/details/elasticnet.hpp
    src/regularization/details/ewc.hpp
    src/regularization/details/fge.hpp
    src/regularization/details/grouplasso.hpp
    src/regularization/details/jacobian.hpp
    src/regularization/details/klsparsity.hpp
    src/regularization/details/l0hardconcrete.hpp
    src/regularization/details/l1.hpp
    src/regularization/details/l2.hpp
    src/regularization/details/mas.hpp
    src/regularization/details/maxnorm.hpp
    src/regularization/details/nuclearnorm.hpp
    src/regularization/details/orthogonality.hpp
    src/regularization/details/r1.hpp
    src/regularization/details/r2.hpp
    src/regularization/details/sfge.hpp
    src/regularization/details/si.hpp
    src/regularization/details/spectralnorm.hpp
    src/regularization/details/structuredl2.hpp
    src/regularization/details/swa.hpp
    src/regularization/details/swag.hpp
    src/regularization/details/trades.hpp
    src/regularization/details/vat.hpp
    src/regularization/details/wgangp.hpp
    src/regularization/regularization.hpp
    src/training/batch_iterator.hpp
    src/training/compute_dataset_loss.hpp
    src/training/dataset_pipeline.hpp
    src/training/deferred_scalar.hpp
    src/training/epoch_runner.hpp
    src/training/graph_coordinator.hpp
    src/training/kfold.hpp
    src/training/step_executor.hpp
    src/training/training_policy.hpp
    src/training/training_preflight.hpp
    src/utils/gnuplot.hpp
    src/utils/progressbar.hpp
    src/utils/terminal.hpp
)
target_link_libraries(Nott PUBLIC "${TORCH_LIBRARIES}" Boost::filesystem ${OpenCV_LIBS})
target_include_directories(Nott PUBLIC ${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# ---- Examples ---------------------------------------------------------------
# One executable target per file under examples/. Every target is registered
# but EXCLUDE_FROM_ALL, so a plain `cmake --build .` builds nothing here --
# each example links the full Torch/OpenCV/Nott stack and building all of
# them unconditionally can take the better part of an hour. Build only the
# one(s) you want, e.g.:
#   cmake --build . --target example_cifar10
# See examples/CMakeLists.txt for the full target list and
# docs/sub/build/README.md for the build guide.
add_subdirectory(examples)
