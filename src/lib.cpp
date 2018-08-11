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

#include "kernelpp/avx_util.h"
#include <mutex>

#ifdef _MSC_VER
# include <intrin.h>
#endif

#if !defined(_XCR_XFEATURE_ENABLED_MASK)
# define _XCR_XFEATURE_ENABLED_MASK 0
#endif

#if defined(_MSC_VER)

void cpuid(uint32_t abcd[4], uint32_t eax) { __cpuid((int*) abcd, eax); }
uint64_t xgetbv(const std::uint32_t xcr) { return _xgetbv(xcr); }

#else

void cpuid(uint32_t abcd[4], uint32_t eax)
{
    uint32_t ecx = 0, ebx = 0, edx = 0;

    __asm__ (
        "cpuid;" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx)
    );

    abcd[0] = eax;
    abcd[1] = ebx;
    abcd[2] = ecx;
    abcd[3] = edx;
}

uint64_t xgetbv(const std::uint32_t xcr)
{
    uint32_t lo, hi;
    __asm__ (
        "xgetbv" : "=a"(lo), "=d"(hi) : "c"(xcr)
    );
    return (static_cast<std::uint64_t>(hi) << 32) | static_cast<std::uint64_t>(lo);
}

#endif

namespace kernelpp
{
    bool init_avx(void)
    {
        static bool success{ false };
        static std::once_flag flag;

        std::call_once(flag, [&]() {
            /* determine whether the CPU supports avx and avx2 */
            uint32_t cpu_info[4] = {0};

            cpuid(cpu_info, 1u);
            bool osUsesXSAVE_XRSTORE = cpu_info[2] & (1 << 27) || false;
            bool cpuAVXSupport = cpu_info[2] & (1 << 28) || false;

            cpuid(cpu_info, 7u);
            bool cpuAVX2Support = cpu_info[1] & (1 << 5) || false;

            if (osUsesXSAVE_XRSTORE && cpuAVXSupport && cpuAVX2Support)
            {
                /* check the OS will save the YMM registers */
                unsigned long long xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
                success = (xcrFeatureMask & 0x6) == 0x6;
            }
        });

        return success;
    }
}
