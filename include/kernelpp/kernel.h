/*  Copyright 2017 International Business Machines Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.  */

#pragma once

#include "kernelpp/config.h"
#include "kernelpp/cuda_util.h"
#include "kernelpp/avx_util.h"

#include <memory>
#include <type_traits>

namespace kernelpp
{
    /* the suppported compute modes */
    enum class compute_mode { AUTO = 1, CUDA, AVX, CPU };

    enum class error_code : uint8_t
    {
        /* The kernel ran and completed successfully */
        NONE = 0,

        /* The kernel was invoked with a compute mode that wasn't
           enabled at compile time  */
        COMPUTE_MODE_DISABLED,

        /* The kernel was invoked with a compute mode unavailable
           at run-time */
        COMPUTE_MODE_UNAVAILABLE,

        /* The kernel does not support being invoked with the
           specified compute mode or arguments */
        KERNEL_NOT_DEFINED,

        /* The kernel was invoked but failed during execution. */
        KERNEL_FAILED,

        /* The kernel was invoked but was cancelled before it
           began executing. */
        CANCELLED
    };

    inline const char* to_str(const error_code s);
    inline const char* to_str(const compute_mode m);


    /*  compute_traits ----------------------------------------------------- */

    template <compute_mode>
    struct compute_traits {
        static constexpr bool enabled = false;
        static bool available() { return false; }
    };

    template <>
    struct compute_traits<compute_mode::CPU> {
        static constexpr bool enabled = true;
        static bool available() { return true; }
    };

#if defined(kernelpp_WITH_AVX)
    template <>
    struct compute_traits<compute_mode::AVX> {
        static constexpr bool enabled = true;
        static bool available() { return kernelpp::init_avx(); }
    };
#endif

#if defined(kernelpp_WITH_CUDA)
    template <>
    struct compute_traits<compute_mode::CUDA> {
        static constexpr bool enabled = true;
        static bool available() { return kernelpp::init_cudart(); }
    };
#endif

    namespace detail
    {
        template <compute_mode T, compute_mode U>
        constexpr bool eq() { return T == U; }

        template <compute_mode... Ts>
        struct has_mode : std::false_type {};

        template <compute_mode T0, compute_mode T1, compute_mode... Ts>
        struct has_mode<T0, T1, Ts...> :
            std::integral_constant<bool, eq<T0, T1>() || has_mode<T0, Ts...>::value>
        {};
    }

    /*  Kernel declarations ------------------------------------------------ */

    template<typename Traits, compute_mode... Modes>
    struct impl
    {
        template <compute_mode M>
        using supports = detail::has_mode<M, Modes... >;

        using traits = Traits;
    };

    #define KERNEL_DECL(Name, ...) \
        struct Name ## _traits_ {                        \
            static constexpr const char* name = #Name;   \
        };                                               \
        struct Name : ::kernelpp::impl<Name ## _traits_, __VA_ARGS__ >


    /*  Implementation detail ---------------------------------------------- */

    inline const char* to_str(const error_code s) {
        switch (s) {
        case error_code::KERNEL_FAILED: return "Kernel Failed";
        case error_code::COMPUTE_MODE_DISABLED: return "Compute Mode Disabled";
        case error_code::COMPUTE_MODE_UNAVAILABLE: return "Compute Mode Unavailable";
        case error_code::KERNEL_NOT_DEFINED: return "Kernel Not Defined";
        case error_code::CANCELLED: return "Cancelled";
        case error_code::NONE: return "Success";
        }
        return "Unknown";
    }

    inline const char* to_str(const compute_mode m) {
        switch (m) {
        case compute_mode::CPU: return "CPU";
        case compute_mode::CUDA: return "Cuda";
        case compute_mode::AVX: return "AVX";
        case compute_mode::AUTO: return "Auto";
        }
        return "Unknown";
    }
}
