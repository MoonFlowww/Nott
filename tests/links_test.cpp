#include "third_party/doctest.h"

#include <torch/torch.h>
#include "../include/Nott.h"

using namespace Nott;

TEST_CASE("links: Broadcast join sums producers (skip connection)") {
    Model model("links_broadcast");
    model.add(Layer::FC({4, 4, false}), "a");
    model.add(Layer::FC({4, 4, false}), "b");

    model.links({
        LinkSpec{Port::Input("@input"), Port::Module("a")},
        LinkSpec{Port::Module("a"), Port::Module("b")},
        LinkSpec{Port::Input("@input"), Port::Join("skip", MergePolicy::Broadcast)},
        LinkSpec{Port::Module("b"), Port::Join("skip", MergePolicy::Broadcast)},
        LinkSpec{Port::Join("skip", MergePolicy::Broadcast), Port::Output("@output")},
    }, /*enable_graph_capture=*/false);

    {
        torch::NoGradGuard guard;
        for (auto& parameter : model.parameters()) {
            parameter.copy_(torch::eye(4));
        }
    }

    auto input = torch::randn({3, 4}, torch::requires_grad(true));
    auto output = model.forward(input);

    REQUIRE(output.defined());
    // a and b are forced to the identity, so b(a(input)) == input; summing that with the
    // raw-input skip branch should give exactly 2*input.
    CHECK(torch::allclose(output, input * 2.0));

    output.sum().backward();
    REQUIRE(input.grad().defined());
    CHECK(torch::isfinite(input.grad()).all().item<bool>());
}

TEST_CASE("links: Stack join concatenates producers along the join dimension") {
    Model model("links_stack");
    model.add(Layer::FC({4, 3, true}), "branch_a");
    model.add(Layer::FC({4, 2, true}), "branch_b");

    model.links({
        LinkSpec{Port::Input("@input"), Port::Module("branch_a")},
        LinkSpec{Port::Input("@input"), Port::Module("branch_b")},
        LinkSpec{Port::Module("branch_a"), Port::Join("concat", MergePolicy::Stack)},
        LinkSpec{Port::Module("branch_b"), Port::Join("concat", MergePolicy::Stack)},
        LinkSpec{Port::Join("concat", MergePolicy::Stack), Port::Output("@output")},
    }, /*enable_graph_capture=*/false);

    auto input = torch::randn({3, 4}, torch::requires_grad(true));
    auto output = model.forward(input);

    REQUIRE(output.defined());
    CHECK(output.sizes() == torch::IntArrayRef({3, 5})); // 3 + 2 concatenated on dim=1
    CHECK(torch::isfinite(output).all().item<bool>());

    output.sum().backward();
    REQUIRE(input.grad().defined());
    CHECK(torch::isfinite(input.grad()).all().item<bool>());
}

TEST_CASE("links: a lone module with no inbound link is rejected at compile time") {
    Model model("links_dangling_module");
    model.add(Layer::FC({4, 4, true}), "a");
    model.add(Layer::FC({4, 4, true}), "orphan"); // deliberately never linked

    CHECK_THROWS_AS(
        model.links({
            LinkSpec{Port::Input("@input"), Port::Module("a")},
            LinkSpec{Port::Module("a"), Port::Output("@output")},
        }, false),
        std::invalid_argument);
}
