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

#pragma once

#include <vector>
#include <memory>
#include "dali_video/decoder/video_decoder_base.h"
#include "dali_video/loader/frames_decoder_gpu.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali_video {

class VideoDecoderMixed
        : public dali::Operator<dali::MixedBackend>, public VideoDecoderBase<dali::MixedBackend, FramesDecoderGpu> {
  using Operator<dali::MixedBackend>::num_threads_;
  using VideoDecoderBase::DecodeSample;

 public:
  explicit VideoDecoderMixed(const dali::OpSpec &spec):
    Operator<dali::MixedBackend>(spec),
    thread_pool_(num_threads_,
                 spec.GetArgument<int>("device_id"),
                 spec.GetArgument<bool>("affine"),
                 "mixed video decoder") {}


  bool CanInferOutputs() const override {
    return true;
  }


  void Run(dali::Workspace &ws) override;

  bool SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                 const dali::Workspace &ws) override;

 private:
  dali::ThreadPool thread_pool_;
};

}  // namespace dali_video
