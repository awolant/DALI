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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_

#include "dali/operators/reader/loader/video/frames_decoder.h"

#include <string>
#include <memory>

#include "dali/operators/reader/loader/video/nvdecode/cuviddec.h"
#include "dali/operators/reader/loader/video/nvdecode/nvcuvid.h"

namespace dali {
struct NvDecodeState {
  CUvideodecoder decoder;
  CUvideoparser parser;

  CUVIDSOURCEDATAPACKET packet = { 0 };

  uint8_t *decoded_frame_yuv;
};

class DLL_PUBLIC FramesDecoderGpu : public FramesDecoder {
 public:
  /**
   * @brief Construct a new FramesDecoder object.
   * 
   * @param filename Path to a video file.
   */
  explicit FramesDecoderGpu(const std::string &filename);

  bool ReadNextFrame(uint8_t *data, bool copy_to_output = true) override;

  void SeekFrame(int frame_id) override;

  void Reset() override;

  std::unique_ptr<NvDecodeState> nvdecode_state_;

  uint8_t *current_frame_output_ = nullptr;
  bool current_copy_to_output_ = false;
  bool decode_success_ = false;
  bool flush_ = false;
  bool last_frame_read_ = false;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_
