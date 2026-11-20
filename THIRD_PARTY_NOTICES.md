<!-- doc-type: product-features -->
# Third-Party Notices & Runtime Permissions

## OpenFX 1.4 SDK & Support Library (BSD-3-Clause)
BetterMotionBlur builds against the OpenFX 1.4 headers and Support Library (`SPDX-License-Identifier: BSD-3-Clause`):

> Copyright (C) 2004-2005 The Open Effects Association Ltd  
> Copyright OpenFX and contributors to the OpenFX project.  
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
> 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
> 3. Neither the name of The Open Effects Association Ltd nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

---

## Runtime-Loaded Libraries (Not Bundled)
- **NVIDIA CUDA Driver (`nvcuda.dll`) & NVRTC (`nvrtc64_120_0.dll`)**: Loaded dynamically at runtime when present on the host system / DaVinci Resolve installation; a full CPU fallback (`CpuFallbackProcessor`) is included when CUDA is unavailable.
- **ONNX Runtime (`onnxruntime.dll`)**: MIT License (Microsoft Corporation). Loaded dynamically at runtime from the host installation when using external `neuflow_v2.onnx` models.
- **GPLv3 Section 7 Additional Permission**: As a special exception under Section 7 of GNU GPLv3, permission is granted to link and load this plugin with OpenFX-compliant host applications and host/driver-provided CUDA, NVRTC, and ONNX Runtime shared libraries.
