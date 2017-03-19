cmake_minimum_required (VERSION 3.0)
cmake_policy (SET CMP0048 NEW) # project versioning

option (kernelpp_WITH_CUDA  "Enable cuda support" OFF)
option (kernelpp_WITH_TESTS "Enable unit tests"   ON)

set (tgt "kernelpp")

project (${tgt}
    LANGUAGES C CXX
    VERSION 0.1.0
)

set (src      "src/lib.cpp")
set (src_cuda "src/lib.cu")

configure_file ("include/kernelpp/config.h.in"
    "include/kernelpp/config.h"
)

if (kernelpp_WITH_CUDA)
    set (CUDA_VERBOSE_BUILD ON)
    find_package (CUDA REQUIRED)
    link_libraries (${CUDA_LIBRARIES})

    CUDA_WRAP_SRCS (${tgt} OBJ
        objs ${src_cuda}

        OPTIONS -cudart static
                --default-stream per-thread
    )
endif ()

add_library (${tgt} STATIC
    ${objs} ${src}
)
target_include_directories (${tgt}
    PUBLIC "third_party/gsl_lite/include"
    PUBLIC "third_party/variant/include"

    PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include"
    PUBLIC "include"
)

if (kernelpp_WITH_TESTS)
    enable_testing ()
    add_subdirectory (test)
endif ()