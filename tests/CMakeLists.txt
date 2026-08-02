# Single test binary: one TU per category, linked once. Torch header
# instantiation dominates compile time here, so we keep the number of test
# executables at one and let CMake parallelize the individual .cpp TUs
# instead of paying separate link/startup overhead per category.
#
# Categories are still separately runnable: each source file below is
# registered as its own ctest entry that filters the shared binary with
# doctest's --source-file, so a failure names the category instead of dumping
# the whole suite. Labels group them for CI (ctest -L <label>).

# Grouped by what breaks together, so a red label points at an area.
set(nott_test_modules
    activation_test.cpp
    loss_test.cpp
    initialization_test.cpp
    regularization_test.cpp
    optimizer_test.cpp
    lrscheduler_test.cpp
    calibration_test.cpp
    attention_kernel_test.cpp
    layer_test.cpp
    block_test.cpp
)

set(nott_test_serialization
    serialize_test.cpp
    model_io_test.cpp
)

set(nott_test_runtime
    links_test.cpp
    precision_test.cpp
    graph_capture_test.cpp
)

set(nott_test_integration
    metric_ground_truth_test.cpp
    training_integration_test.cpp
)

add_executable(nott_tests EXCLUDE_FROM_ALL
    main.cpp
    ${nott_test_modules}
    ${nott_test_serialization}
    ${nott_test_runtime}
    ${nott_test_integration}
)
target_link_libraries(nott_tests PRIVATE Nott)
target_include_directories(nott_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(CTest)

# One ctest entry per source file. The category name is derived from the file
# name rather than written twice, so a rename cannot leave a filter silently
# matching nothing (doctest exits 0 when a filter selects no test cases).
function(nott_register_test_category source label)
    get_filename_component(category "${source}" NAME_WE)
    string(REGEX REPLACE "_test$" "" category "${category}")
    add_test(NAME ${category}
             COMMAND nott_tests --source-file=*${source} --no-skipped-summary)
    set_tests_properties(${category} PROPERTIES LABELS "${label}")
endfunction()

foreach(source IN LISTS nott_test_modules)
    nott_register_test_category(${source} "module")
endforeach()
foreach(source IN LISTS nott_test_serialization)
    nott_register_test_category(${source} "serialization")
endforeach()
foreach(source IN LISTS nott_test_runtime)
    nott_register_test_category(${source} "runtime")
endforeach()
foreach(source IN LISTS nott_test_integration)
    nott_register_test_category(${source} "integration")
endforeach()
