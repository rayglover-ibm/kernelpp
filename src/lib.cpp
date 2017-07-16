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
#  include <intrin.h>
#endif

#ifdef __GNUC__
void __cpuid(int* cpuinfo, int info) {
    __asm__ __volatile__(
        "xchg %%ebx, %%edi;"
        "cpuid;"
        "xchg %%ebx, %%edi;"
        :"=a" (cpuinfo[0]), "=D" (cpuinfo[1]), "=c" (cpuinfo[2]), "=d" (cpuinfo[3])
        :"0" (info)
    );
}

unsigned long long _xgetbv(unsigned int index) {
    unsigned int eax, edx;
    __asm__ __volatile__(
        "xgetbv;"
        : "=a" (eax), "=d"(edx)
        : "c" (index)
    );
    return ((unsigned long long)edx << 32) | eax;
}
#endif

namespace kernelpp
{
    bool init_avx(void)
    {
        static bool success{ false };
        static std::once_flag flag;

        std::call_once(flag, [&]() {
            int cpuinfo[4];
            __cpuid(cpuinfo, 1);

            bool avxSupportted    = cpuinfo[2] & (1 << 28) || false;
            bool osxsaveSupported = cpuinfo[2] & (1 << 27) || false;

            if (osxsaveSupported && avxSupportted)
            {
                unsigned long long xcrFeatureMask = _xgetbv(0);
                avxSupportted = (xcrFeatureMask & 0x6) == 0x6;
            }

            success = avxSupportted;
        });

        return success;
    }
}