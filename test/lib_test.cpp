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

#include "gtest/gtest.h"

#include "kernelpp/kernel.h"
#include "kernelpp/kernel_invoke.h"
#include "kernelpp/avx_util.h"

#include <array>
#include <vector>

using namespace kernelpp;

namespace
{
    template <compute_mode M>
    void print_compute_info()
    {
        using traits = compute_traits<M>;
        std::cout << to_str(M) << " enabled: " << traits::enabled << ", available: "
            << (traits::enabled ? traits::available() : 0) << std::endl;
    }
}

TEST(kernel, compute_mode_info)
{
    print_compute_info<compute_mode::CPU>();
    print_compute_info<compute_mode::AVX>();
    print_compute_info<compute_mode::CUDA>();
}

/* ------------------------------------------------------------------------- */

namespace
{
    KERNEL_DECL(foo, compute_mode::CPU)
    {
        template<compute_mode> static void op();
        template<compute_mode> static int  op(std::vector<float>&);
    };

    int void_calls = 0;

    template <> void foo::op<compute_mode::CPU>() {
        void_calls++;
    }

    template <> int foo::op<compute_mode::CPU>(std::vector<float>& v) {
        return (int)v.size();
    }
}

TEST(kernel, call_void)
{
    ::void_calls = 0;
    status err = run<foo>();

    EXPECT_FALSE(err);
    EXPECT_EQ(::void_calls, 1);

    err = run<foo, compute_mode::CPU>();

    EXPECT_FALSE(err);
    EXPECT_EQ(::void_calls, 2);
}

TEST(kernel, call_undefined)
{
    ::void_calls = 0;
    status err = run<foo, compute_mode::CUDA>();

    EXPECT_TRUE(err);
    EXPECT_EQ(::void_calls, 0);
}

TEST(kernel, call_vector)
{
    ::void_calls = 0;

    std::vector<float> vec(5, 0);
    maybe<int> result = run<foo>(vec);

    EXPECT_FALSE(result.is<error>());
    EXPECT_EQ(result.get<int>(), 5);
}

/* cuda -------------------------------------------------------------------- */

namespace
{
    KERNEL_DECL(foo_2, compute_mode::CPU, compute_mode::CUDA) {
        template<compute_mode> static void op();
    };

    int cuda_calls = 0;
    int cpu_calls = 0;

    template <> void foo_2::op<compute_mode::CPU>()  { cpu_calls++; }
    template <> void foo_2::op<compute_mode::CUDA>() { cuda_calls++; }
}

TEST(kernel, call_cuda)
{
    using cuda = compute_traits<compute_mode::CUDA>;

    cuda_calls = 0;
    cpu_calls = 0;

    EXPECT_FALSE(run<foo_2>());

    if (cuda::enabled && cuda::available()) {
        EXPECT_EQ(0, cpu_calls);
        EXPECT_EQ(1, cuda_calls);
    }
    else {
        EXPECT_EQ(1, cpu_calls);
        EXPECT_EQ(0, cuda_calls);
    }
}

/* avx --------------------------------------------------------------------- */

namespace
{
    KERNEL_DECL(foo_3, compute_mode::CPU, compute_mode::AVX)
    {
        template<compute_mode> static void op();
        template<compute_mode> static error_code op(std::vector<float>&);
    };

    int avx_calls = 0;

    template <> void foo_3::op<compute_mode::CPU>() { cpu_calls++; }
    template <> void foo_3::op<compute_mode::AVX>() { avx_calls++; }
}

TEST(kernel, call_avx)
{
    using avx = compute_traits<compute_mode::AVX>;

    avx_calls = 0;
    cpu_calls = 0;

    EXPECT_FALSE(run<foo_3>());

    if (avx::enabled && avx::available()) {
        EXPECT_EQ(0, cpu_calls);
        EXPECT_EQ(1, avx_calls);
    }
    else {
        EXPECT_EQ(1, cpu_calls);
        EXPECT_EQ(0, avx_calls);
    }
}

TEST(avx_util, is_aligned)
{
    using T = int32_t;

    EXPECT_TRUE((kernelpp::is_aligned<T, 1>(nullptr)));
    EXPECT_TRUE((kernelpp::is_aligned<T, 1>(reinterpret_cast<T*>(0x00))));
    EXPECT_TRUE((kernelpp::is_aligned<T, 1>(reinterpret_cast<T*>(0x04))));
    EXPECT_TRUE((kernelpp::is_aligned<T, 2>(reinterpret_cast<T*>(0x08))));

    EXPECT_FALSE((kernelpp::is_aligned<T, 1>(reinterpret_cast<T*>(0x01))));
    EXPECT_FALSE((kernelpp::is_aligned<T, 1>(reinterpret_cast<T*>(0x02))));
    EXPECT_FALSE((kernelpp::is_aligned<T, 1>(reinterpret_cast<T*>(0x03))));
    EXPECT_FALSE((kernelpp::is_aligned<T, 2>(reinterpret_cast<T*>(0x04))));
}

/* extras ------------------------------------------------------------------ */

TEST(runners, log_runner)
{
    log_runner<foo> r(&std::cout);
    EXPECT_FALSE(run_with<foo>(r));
}

/* nested kernels ---------------------------------------------------------- */

namespace
{
    struct foo_struct
    {
        int calls = 0;
        void call() { run<kern_a>(this); }

      private:
        KERNEL_DECL(kern_a, compute_mode::CPU) {
            template<compute_mode> static void op(foo_struct* state) {
                state->calls++;
            }
        };
    };
}

TEST(struct_example, call)
{
    foo_struct f;
    f.call();

    EXPECT_EQ(1, f.calls);
}

/* specializations --------------------------------------------------------- */

KERNEL_DECL(kern_special, compute_mode::CPU, compute_mode::AVX)
{
    template <compute_mode M> struct tag {};

    static int avx_i32;
    static int avx_f32;
    static int cpu;

    /* generic cpu impl */
    template <typename T>
    static void foo(T val, tag<compute_mode::CPU>)     { cpu++; }
    
    /* avx float/int impl */
    static void foo(float val, tag<compute_mode::AVX>) { avx_f32++; }
    static void foo(int val, tag<compute_mode::AVX>)   { avx_i32++; }

    template <compute_mode M, typename T>
    static void op(T val) { foo(val, tag<M>{}); }
};

int kern_special::avx_i32 = 0;
int kern_special::avx_f32 = 0;
int kern_special::cpu = 0;

TEST(specializations, call_generic)
{
    run<kern_special, compute_mode::CPU>(float(1.0));
    run<kern_special, compute_mode::AVX>(float(1));
    run<kern_special, compute_mode::AVX>(int(1));

    EXPECT_EQ(1, kern_special::cpu);
    EXPECT_EQ(1, kern_special::avx_i32);
    EXPECT_EQ(1, kern_special::avx_f32);
}