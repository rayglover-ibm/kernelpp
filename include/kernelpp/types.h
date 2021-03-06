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

#include <mapbox/variant.hpp>
#include <mapbox/optional.hpp>

namespace kernelpp
{
    /* Types  -------------------------------------------------------------- */

    /* standard kernelpp error type */
    using error = std::string;

    /* introduce a variant type. Replace with std::variant
       once C++17 is well supported. */
    using mapbox::util::variant;

    /* an optional value of type T or an error */
    template <typename T> using maybe = variant<T, error>;

    /* an optional error */
    using status = mapbox::util::optional<error>;
}