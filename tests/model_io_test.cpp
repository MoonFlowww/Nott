#include "third_party/doctest.h"

#include <torch/torch.h>
#include <filesystem>
#include "../include/Nott.h"

using namespace Nott;

TEST_CASE("model_io: save/load round-trips a single-layer model's parameters") {
    Model model("model_io_single_layer");
    model.add(Layer::FC({4, 3, true}), "fc");

    auto input = torch::randn({2, 4});
    auto before = model.forward(input);

    auto scratch = std::filesystem::temp_directory_path() / "nott_model_io_test_single";
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    model.save(scratch);

    Model reloaded("model_io_single_layer_target");
    reloaded.load(scratch / "model_io_single_layer");
    auto after = reloaded.forward(input);

    CHECK(torch::allclose(before, after));
    std::filesystem::remove_all(scratch);
}

TEST_CASE("model_io: save/load round-trips a multi-layer sequential model") {
    Model model("model_io_multi_layer");
    model.add(Layer::FC({4, 8, true}), "fc1");
    model.add(Layer::FC({8, 3, true}), "fc2");

    auto input = torch::randn({5, 4});
    auto before = model.forward(input);

    auto scratch = std::filesystem::temp_directory_path() / "nott_model_io_test_multi";
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    model.save(scratch);

    Model reloaded("model_io_multi_layer_target");
    reloaded.load(scratch / "model_io_multi_layer");
    auto after = reloaded.forward(input);

    CHECK(torch::allclose(before, after));
    std::filesystem::remove_all(scratch);
}
