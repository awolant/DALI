// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_IMGCODEC_PARSERS_JPEG2000_H_
#define DALI_IMGCODEC_PARSERS_JPEG2000_H_

#include "dali/imgcodec/image_format.h"

namespace dali {
namespace imgcodec {

class DLL_PUBLIC Jpeg2000Parser : public ImageParser {
 public:
  ImageInfo GetInfo(ImageSource *encoded) const override;
  bool CanParse(ImageSource *encoded) const override;
};

#endif  // DALI_IMGCODEC_PARSERS_JPEG2000_H_

}  // namespace imgcodec
}  // namespace dali
