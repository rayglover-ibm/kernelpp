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


#ifdef __amd64__
# define _XCR_XFEATURE_ENABLED_MASK 0

void __cpuid(uint32_t abcd[4], uint32_t eax)
{
    /* Reference: https://software.intel.com/en-us/articles/
           how-to-detect-new-instruction-support-in-the-4th-generation-
           intel-core-processor-family
    */
    uint32_t ecx = 0, ebx = 0, edx = 0;

# if defined(_MSC_VER)
    __cpuidex(abcd, eax, ecx);
# else
    __asm__ (
        "cpuid;" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx)
    );
# endif

    abcd[0] = eax;
    abcd[1] = ebx;
    abcd[2] = ecx;
    abcd[3] = edx;
}

uint64_t _xgetbv(const std::uint32_t xcr)
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
            uint32_t cpu_info[4] = {0};

            __cpuid(cpu_info, 1u);
            bool osUsesXSAVE_XRSTORE = cpu_info[2] & (1 << 27) || false;
            bool cpuAVXSupport = cpu_info[2] & (1 << 28) || false;

            __cpuid(cpu_info, 7u);
            bool cpuAVX2Support = cpu_info[1] & (1 << 5) || false;

            if (osUsesXSAVE_XRSTORE && cpuAVXSupport && cpuAVX2Support)
            {
                unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
                success = (xcrFeatureMask & 0x6) == 0x6;
            }
        });

        return success;
    }
}