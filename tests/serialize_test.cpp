/// Descriptor level save/load round-trips. model_io_test covers tensor state;
/// this covers the JSON description of the architecture, which is what the
/// per module serialize.hpp files produce.
#include "third_party/doctest.h"

#include "../src/common/save_load.hpp"

namespace {
    namespace SaveLoad = Nott::Common::SaveLoad;
    namespace Transformer = Nott::Block::Details::Transformer;

    /// Round-trips through the same text form Model::save writes, so a mismatch
    /// between what a descriptor writes and what it reads shows up here.
    template <class Descriptor, class Serialize, class Deserialize>
    Descriptor round_trip(const Descriptor& descriptor, Serialize serialize, Deserialize deserialize)
    {
        std::ostringstream text;
        boost::property_tree::write_json(text, serialize(descriptor), false);
        std::istringstream input(text.str());
        Nott::Serialize::PropertyTree tree;
        boost::property_tree::read_json(input, tree);
        return deserialize(tree, "test");
    }

    Nott::Layer::Descriptor layer_round_trip(const Nott::Layer::Descriptor& descriptor)
    {
        return round_trip(descriptor, Nott::Layer::serialize_layer_descriptor,
                          Nott::Layer::deserialize_layer_descriptor);
    }

    Nott::Block::Descriptor block_round_trip(const Nott::Block::Descriptor& descriptor)
    {
        return round_trip(descriptor, Nott::Block::serialize_block_descriptor,
                          Nott::Block::deserialize_block_descriptor);
    }
}

TEST_CASE("layer descriptors round-trip through json") {
    Nott::Layer::FCOptions fc_options;
    fc_options.in_features = 16;
    fc_options.out_features = 4;
    fc_options.bias = false;
    const auto restored = layer_round_trip(Nott::Layer::FC(fc_options, Nott::Activation::ReLU));
    const auto& fc = std::get<Nott::Layer::Details::FCDescriptor>(restored);
    CHECK(fc.options.in_features == 16);
    CHECK(fc.options.out_features == 4);
    CHECK(fc.options.bias == false);
    CHECK(fc.activation.type == Nott::Activation::Type::ReLU);
}

TEST_CASE("pooling keeps its options alternative across a round-trip") {
    Nott::Layer::AvgPool2dOptions options;
    options.kernel_size = {3, 3};
    options.stride = {2, 2};
    options.padding = {1, 1};
    options.count_include_pad = false;
    const auto restored = layer_round_trip(Nott::Layer::AvgPool2d(options));
    const auto& pooling = std::get<Nott::Layer::Details::PoolingDescriptor>(restored);
    const auto& avg = std::get<Nott::Layer::Details::AvgPool2dOptions>(pooling.options);
    const std::vector<std::int64_t> expected_kernel{3, 3};
    CHECK(avg.kernel_size == expected_kernel);
    CHECK(avg.count_include_pad == false);
}

TEST_CASE("sequential block round-trips its layers") {
    Nott::Layer::FCOptions fc_options;
    fc_options.in_features = 8;
    fc_options.out_features = 2;
    const auto restored = block_round_trip(Nott::Block::Sequential({Nott::Layer::FC(fc_options)}));
    const auto& sequential = std::get<Nott::Block::Details::SequentialDescriptor>(restored);
    REQUIRE(sequential.layers.size() == 1);
    CHECK(std::get<Nott::Layer::Details::FCDescriptor>(sequential.layers.front()).options.out_features == 2);
}

/// These three families used to serialize with no matching reader, so saving a
/// model containing one produced a file that threw on load.
TEST_CASE("vision encoder round-trips") {
    Transformer::Vision::EncoderDescriptor descriptor;
    descriptor.options.layers = 2;
    descriptor.options.embed_dim = 32;
    descriptor.options.variant = Transformer::Vision::Variant::Swin;
    descriptor.options.window.size = 7;
    descriptor.options.window.shift = true;
    descriptor.layers.emplace_back();

    const auto& vision = std::get<Transformer::Vision::EncoderDescriptor>(
        block_round_trip(Nott::Block::Descriptor{descriptor}));
    CHECK(vision.options.layers == 2);
    CHECK(vision.options.embed_dim == 32);
    CHECK(vision.options.variant == Transformer::Vision::Variant::Swin);
    CHECK(vision.options.window.size == 7);
    CHECK(vision.options.window.shift == true);
    CHECK(vision.layers.size() == 1);
}

TEST_CASE("perceiver encoder round-trips") {
    Transformer::Perceiver::EncoderDescriptor descriptor;
    descriptor.options.layers = 3;
    descriptor.options.latent_slots = 64;
    descriptor.options.latent_dim = 128;
    descriptor.layers.emplace_back();

    const auto& perceiver = std::get<Transformer::Perceiver::EncoderDescriptor>(
        block_round_trip(Nott::Block::Descriptor{descriptor}));
    CHECK(perceiver.options.layers == 3);
    CHECK(perceiver.options.latent_slots == 64);
    CHECK(perceiver.options.latent_dim == 128);
    CHECK(perceiver.layers.size() == 1);
}

TEST_CASE("longformer encoder round-trips") {
    Transformer::LongformerXL::EncoderDescriptor descriptor;
    descriptor.options.layers = 2;
    descriptor.options.embed_dim = 64;
    descriptor.options.window_size = 128;
    descriptor.options.causal = true;
    descriptor.options.use_memory = true;
    descriptor.options.memory_size = 16;
    descriptor.layers.emplace_back();

    const auto& longformer = std::get<Transformer::LongformerXL::EncoderDescriptor>(
        block_round_trip(Nott::Block::Descriptor{descriptor}));
    CHECK(longformer.options.window_size == 128);
    CHECK(longformer.options.causal == true);
    CHECK(longformer.options.memory_size == 16);
    CHECK(longformer.layers.size() == 1);
}

/// adafactor read use_first_moment and weight_decay with each other's types,
/// and lamb/lion looked for options.lr while writing options.learning_rate.
/// All three threw on any load.
TEST_CASE("optimizers that could not be loaded now round-trip") {
    const auto optimizer_round_trip = [](const Nott::Optimizer::Descriptor& descriptor) {
        return round_trip(descriptor, Nott::Optimizer::serialize_optimizer,
                          Nott::Optimizer::deserialize_optimizer);
    };

    SUBCASE("adafactor") {
        Nott::Optimizer::Details::AdafactorOptions options;
        options.use_first_moment(true);
        options.weight_decay(0.25);
        const auto restored = optimizer_round_trip(
            Nott::Optimizer::Descriptor{Nott::Optimizer::Details::AdafactorDescriptor{options}});
        const auto& adafactor = std::get<Nott::Optimizer::Details::AdafactorDescriptor>(restored);
        CHECK(adafactor.options.use_first_moment() == true);
        CHECK(adafactor.options.weight_decay() == doctest::Approx(0.25));
    }

    SUBCASE("lamb") {
        Nott::Optimizer::Details::LAMBOptions options;
        options.lr(0.007);
        const auto restored =
            optimizer_round_trip(Nott::Optimizer::Descriptor{Nott::Optimizer::Details::LAMBDescriptor{options}});
        CHECK(std::get<Nott::Optimizer::Details::LAMBDescriptor>(restored).options.lr() == doctest::Approx(0.007));
    }

    SUBCASE("lion") {
        Nott::Optimizer::Details::LionOptions options;
        options.lr(0.003);
        const auto restored =
            optimizer_round_trip(Nott::Optimizer::Descriptor{Nott::Optimizer::Details::LionDescriptor{options}});
        CHECK(std::get<Nott::Optimizer::Details::LionDescriptor>(restored).options.lr() == doctest::Approx(0.003));
    }
}

TEST_CASE("activation survives as a name, and the old index form still loads") {
    Transformer::Bert::FeedForwardOptions options;
    options.embed_dim = 16;
    options.mlp_ratio = 4.0;
    options.bias = true;
    options.activation = Nott::Activation::GeLU;

    const auto tree = Transformer::Bert::serialize_feed_forward_options(options);
    CHECK(tree.get<std::string>("activation.type") == "gelu");
    CHECK(Transformer::Bert::deserialize_feed_forward_options(tree, "test").activation.type ==
          Nott::Activation::Type::GeLU);

    /// A file written before activations were named stores the bare enum index.
    auto legacy = tree;
    legacy.put("activation.type", static_cast<std::uint64_t>(Nott::Activation::Type::Mish));
    CHECK(Transformer::Bert::deserialize_feed_forward_options(legacy, "test").activation.type ==
          Nott::Activation::Type::Mish);
}

TEST_CASE("an unknown type name is reported rather than silently accepted") {
    Nott::Serialize::PropertyTree tree;
    tree.put("type", "not_a_layer");
    CHECK_THROWS_AS(Nott::Layer::deserialize_layer_descriptor(tree, "test"), std::runtime_error);
}
