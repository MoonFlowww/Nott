#define DOCTEST_CONFIG_IMPLEMENT
#include "test_prelude.hpp"

int main(int argc, char** argv) {
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}
