#include "third_party/doctest.h"

#include <torch/torch.h>
#include "../include/Nott.h"

using namespace Nott;

// set_precision is now the single place that touches process-global TF32/cuDNN-benchmark state

TEST_CASE("precision: set_precision flips the global TF32 flags") {
    Model model("precision_smoke");

    model.set_precision(/*allow_tf32=*/false);
    CHECK_FALSE(at::globalContext().allowTF32CuDNN());
    CHECK_FALSE(at::globalContext().allowTF32CuBLAS());

    model.set_precision(/*allow_tf32=*/true);
    CHECK(at::globalContext().allowTF32CuDNN());
    CHECK(at::globalContext().allowTF32CuBLAS());
}

TEST_CASE("precision: constructing LSTM/GRU layers no longer touches global TF32 state") {
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowTF32CuBLAS(false);

    Model model("precision_no_side_effect");
    model.add(Layer::LSTM({.input_size = 4, .hidden_size = 4, .batch_first = true}));
    model.add(Layer::GRU({.input_size = 4, .hidden_size = 4, .batch_first = true}));

    CHECK_FALSE(at::globalContext().allowTF32CuDNN());
    CHECK_FALSE(at::globalContext().allowTF32CuBLAS());
}
