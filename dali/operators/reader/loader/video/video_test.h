// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_TEST_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_TEST_H_

#include <gtest/gtest.h>
#include <opencv2/core.hpp>


namespace dali {
class VideoTest : public ::testing::Test {
 public:
  VideoTest();

  const int NumVideos() const { return gt_frames_.size(); }

  const int NumFrames(int i) const { return gt_frames_[i].size(); }

  const int Channels() const { return 3; }

  const int Width(int i) const { return gt_frames_[i][0].cols; }

  const int Height(int i) const { return gt_frames_[i][0].rows; }

  const int FrameSize(int i) const { return Height(i) * Width(i) * Channels(); }

  void ComapreFrames(const uint8_t *frame, const uint8_t *gt, size_t size, int eps = 0);

  void SaveFrame(uint8_t *frame, int frame_id, int sample_id, int batch_id, std::string subfolder, int width, int height, int channels);

 protected:
  std::vector<std::vector<cv::Mat>> gt_frames_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_LOADER_CPU_H_
