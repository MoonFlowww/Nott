# Single test binary: one TU per category, linked once. Torch header
# instantiation dominates compile time here, so we keep the number of test
# executables at one and let CMake parallelize the individual .cpp TUs
# instead of paying separate link/startup overhead per category.
add_executable(nott_tests EXCLUDE_FROM_ALL
    main.cpp
    activation_test.cpp
    loss_test.cpp
    initialization_test.cpp
    regularization_test.cpp
    optimizer_test.cpp
    attention_kernel_test.cpp
    layer_test.cpp
    precision_test.cpp
    block_test.cpp
    links_test.cpp
    graph_capture_test.cpp
    model_io_test.cpp
    metric_ground_truth_test.cpp
    training_integration_test.cpp
)
target_link_libraries(nott_tests PRIVATE Nott)
target_include_directories(nott_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(CTest)
add_test(NAME nott_tests COMMAND nott_tests)
