# One executable per example source file, each linked against the Nott
# umbrella target (headers + Torch/Boost/OpenCV). Every target below is
# EXCLUDE_FROM_ALL: nothing here is built by `cmake --build .` on its own --
# name the target(s) you actually want, e.g.:
#   cmake --build . --target example_cifar10
#   cmake --build . --target example_etth example_ptbxl
set(NOTT_EXAMPLES
    example_ntk           analysis/ntk.cpp
    example_overhead      analysis/overhead.cpp
    example_cifar10       classification/images/cifar10.cpp
    example_dubai_segment classification/masks/dubai_segment.cpp
    example_ptbxl         classification/timeseries/ptbxl.cpp
    example_etth          regression/timeseries/etth.cpp
    example_speedtest     speedtest.cpp
)

list(LENGTH NOTT_EXAMPLES _nott_examples_count)
math(EXPR _nott_examples_last "${_nott_examples_count} - 1")
foreach(_i RANGE 0 ${_nott_examples_last} 2)
    list(GET NOTT_EXAMPLES ${_i} _nott_example_target)
    math(EXPR _j "${_i} + 1")
    list(GET NOTT_EXAMPLES ${_j} _nott_example_source)

    add_executable(${_nott_example_target} EXCLUDE_FROM_ALL ${_nott_example_source})
    target_link_libraries(${_nott_example_target} PRIVATE Nott)
endforeach()
unset(_nott_examples_count)
unset(_nott_examples_last)
unset(_i)
unset(_j)
unset(_nott_example_target)
unset(_nott_example_source)
