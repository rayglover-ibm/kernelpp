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

#include "kernelpp/types.h"
#include "kernelpp/kernel.h"

#include <memory>
#include <type_traits>
#include <ostream>

namespace kernelpp
{
    /* Kernel output traits ------------------------------------------------ */

    namespace detail
    {
        template <typename R> struct op_traits
        {
            using output_type = variant<R, error_code>;
            using public_type = maybe<R>;

            static constexpr bool is_void = false;
            static error_code get_errc(const output_type& s)
            {
                return s.template is<error_code>() ?
                    s.template get<error_code>() : error_code::NONE;
            }
        };
        template <typename R> struct op_traits<variant<R, error_code>>
        {
            using output_type = variant<R, error_code>;
            using public_type = maybe<R>;

            static constexpr bool is_void = false;
            static error_code get_errc(const output_type& s)
            {
                return s.template is<error_code>() ?
                    s.template get<error_code>() : error_code::NONE;
            }
        };
        template <> struct op_traits<void>
        {
            using output_type = error_code;
            using public_type = status;

            static constexpr bool is_void = true;
            static error_code get_errc(const output_type& s) { return s; }
        };
        template <> struct op_traits<error_code>
        {
            using output_type = error_code;
            using public_type = status;

            static constexpr bool is_void = false;
            static error_code get_errc(const output_type& s) { return s; }
        };

        template <typename K, typename... Args>
        struct op_trait_helper
        {
            using type =
                typename detail::op_traits<
                    decltype(K::template op<compute_mode::AUTO>(std::declval<Args>()...))
                    >;
        };

        template <typename K>
        struct op_trait_helper<K>
        {
            using type =
                typename detail::op_traits<
                    decltype(K::template op<compute_mode::AUTO>())
                    >;
        };
    }

    template <typename K, typename... Args>
    using op_traits = typename detail::op_trait_helper<K, Args...>::type;

    template <typename K, typename... Args>
    using result = typename op_traits<K, Args...>::output_type;


    /*  Runtime kernel selection ------------------------------------------- */

    template <compute_mode M, typename = void>
    struct control;

    /*  Used when the compute_mode is enabled at compilation-time  */
    template <compute_mode M>
    struct control<M, std::enable_if_t< compute_traits<M>::enabled >>
    {
        template <typename K, typename Runner, typename... Args>
        static auto call(Runner& r, Args&&... args)
            -> result<K, Args...>
        {
            if (!r.begin(M)) { return error_code::CANCELLED; }
            result<K, Args...> s = r.template apply<M>(std::forward<Args>(args)...);
            r.end(op_traits<K, Args...>::get_errc(s));

            return s;
        }
    };

    /*  Used when the compute_mode isn't enabled at compilation-time  */
    template <compute_mode M>
    struct control<M, std::enable_if_t< !compute_traits<M>::enabled >>
    {
        template <typename Kernel, typename Runner, typename... Args>
        static auto call(Runner&, Args&&...)
            -> result<Kernel, Args...>
        {
            return error_code::COMPUTE_MODE_DISABLED;
        }
    };

    /*  Specialization for AUTO: determines compute_mode at runtime  */
    template <>
    template <typename Kernel, typename Runner, typename... Args>
    auto control<compute_mode::AUTO>::call(Runner& r, Args&&... args)
        -> result<Kernel, Args...>
    {
        result<Kernel, Args...> s = error_code::KERNEL_NOT_DEFINED;

        /* Attempt to run cuda kernel */
        if (compute_traits<compute_mode::CUDA>::available())
        {
            s = control<compute_mode::CUDA>::call<Kernel>(
                    r, std::forward<Args>(args)...);
        }
        /* Attempt to run avx kernel */
        if (s != error_code::KERNEL_NOT_DEFINED &&
            compute_traits<compute_mode::AVX>::available())
        {
            s = control<compute_mode::AVX>::call<Kernel>(
                    r, std::forward<Args>(args)...);
        }
        /* Attempt/fallback to run cpu kernel */
        if (s != error_code::KERNEL_NOT_DEFINED &&
            compute_traits<compute_mode::CPU>::available())
        {
            s = control<compute_mode::CPU>::call<Kernel>(
                    r, std::forward<Args>(args)...);
        }
        return s;
    }

    /*  kernel runner ------------------------------------------------------ */

    template <typename K>
    struct runner
    {
        using traits = typename K::traits;

        bool begin(compute_mode) { return true; }
        void end(error_code) {}

        /* when the kernel doesnt support the given compute_mode */
        template <
            compute_mode M, typename... Args,
            std::enable_if_t< !K::template supports<M>::value, int> = 0
            >
        auto apply(Args&&... args) -> result<K, Args...> {
            return error_code::KERNEL_NOT_DEFINED;
        }

        /* when the kernel's return type is non-void */
        template <
            compute_mode M, typename... Args,
            std::enable_if_t<
                K::template supports<M>::value &&
                !op_traits<K, Args...>::is_void, int
                > = 0
            >
        auto apply(Args&&... args) -> result<K, Args...> {
            return K::template op<M>(std::forward<Args>(args)...);
        }

        /* when the kernel's return type is void */
        template <
            compute_mode M, typename... Args,
            std::enable_if_t<
                K::template supports<M>::value &&
                op_traits<K, Args...>::is_void, int
                > = 0
            >
        auto apply(Args&&... args) -> result<K, Args...>
        {
            K::template op<M>(std::forward<Args>(args)...);
            return error_code::NONE;
        }
    };



    /*  public api --------------------------------------------------------- */

    namespace detail
    {
        template <typename R>
        maybe<R> convert(variant<R, error_code>&& r)
        {
            return r.match(
                [](error_code s) -> maybe<R> { return to_str(s); },
                [](R& result)    -> maybe<R> { return std::move(result); }
                );
        }

        inline status convert(error_code r) {
            return r == error_code::NONE ? status() : status{ to_str(r) };
        }
    }

    template <
        typename K,
        compute_mode M = compute_mode::AUTO,
        typename Runner,
        typename... Args
        >
    typename op_traits<K, Args...>::public_type run_with(
        Runner& r, Args&&... args)
    {
        return detail::convert(
            control<M>::template call<K>(r, std::forward<Args>(args)...)
            );
    }

    template <
        typename K,
        compute_mode M = compute_mode::AUTO,
        typename... Args
        >
    typename op_traits<K, Args...>::public_type run(
        Args&&... args)
    {
        runner<K> r;

        return detail::convert(
            control<M>::template call<K>(r, std::forward<Args>(args)...)
            );
    }


    /* Extensions ---------------------------------------------------------- */

    template <typename K>
    struct log_runner : public runner<K>
    {
        using typename runner<K>::traits;
        log_runner(std::ostream* out) : m_out(out) {}

        bool begin(compute_mode m)
        {
            *m_out << "[" << traits::name << "] mode="
                   << to_str(m) << std::endl;

            return true;
        }

        void end(error_code s)
        {
            *m_out << "[" << traits::name << "] status="
                   << to_str(s) << std::endl;
        }

        private: std::ostream* m_out;
    };
}