cmake_minimum_required (VERSION 3.0)
cmake_policy (SET CMP0048 NEW) # project versioning

# options ---------------------------------------------------------------------
option (kernelpp_WITH_CUDA  "Enable cuda support" OFF)
option (kernelpp_WITH_AVX   "Enable avx support"  ON)
option (kernelpp_WITH_TESTS "Enable unit tests"   ON)
# -----------------------------------------------------------------------------

set (tgt "kernelpp")

project (${tgt}
    LANGUAGES C CXX
    VERSION 0.1.0
)

set (src      "src/lib.cpp")
set (src_cuda "src/lib.cu")

# language requirements/compiler opts
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# configurable header
configure_file (
    "include/kernelpp/config.h.in"
    "include/kernelpp/config.h"
)
list (APPEND inc
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gsl_lite/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/variant/include"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# main library
add_library (${tgt} STATIC ${src})
target_include_directories (${tgt} PUBLIC ${inc})

if (kernelpp_WITH_CUDA)
    # TODO(rayg): revise once CMake 3.8 is released
    set (CUDA_VERBOSE_BUILD ON)
    find_package (CUDA REQUIRED)

    if (UNIX AND CMAKE_POSITION_INDEPENDENT_CODE)
        list (APPEND CMAKE_CXX_FLAGS "-fPIC")
    endif ()

    link_libraries (${CUDA_LIBRARIES})
    cuda_include_directories (${inc})
    cuda_wrap_srcs (${tgt} OBJ
        obj_cuda ${src_cuda}

        OPTIONS -cudart static
                --default-stream per-thread
                -std=c++11
    )
    target_sources (${tgt} PRIVATE ${obj_cuda})
endif ()

# tests
if (kernelpp_WITH_TESTS)
    enable_testing ()
    add_subdirectory (test)
endif ()
