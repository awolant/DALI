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

#include "dali/operators/reader/loader/video/video_test_base.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <thread>

#include "dali/test/dali_test_config.h"
#include "dali/test/cv_mat_utils.h"

namespace dali {
namespace detail {
static void parallel_for(
  unsigned nb_elements, std::function<void (int start, int end, int id)> functor) {
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);
    unsigned batch_size = nb_elements / nb_threads;
    unsigned batch_remainder = nb_elements % nb_threads;

    std::vector< std::thread > my_threads(nb_threads);
    for(unsigned i = 0; i < nb_threads; ++i)
    {
        int start = i * batch_size;
        my_threads[i] = std::thread(functor, start, start+batch_size, i);
    }

    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}
} 

// Define static tests members - needed to hold resources between tests
std::vector<std::vector<cv::Mat>> VideoTestBase::cfr_frames_;
std::vector<std::vector<cv::Mat>> VideoTestBase::vfr_frames_;

void VideoTestBase::SetUpTestSuite() {
  if (cfr_frames_.size() > 0) {
    // This setup is called for each test fixture, but we need it only once
    return;
  }
  std::vector<std::string> cfr_frames_paths{
      testing::dali_extra_path() + "/db/video/cfr/frames_1/",
      testing::dali_extra_path() + "/db/video/cfr/frames_2/"};

  std::vector<std::string> vfr_frames_paths{
      testing::dali_extra_path() + "/db/video/vfr/frames_1/",
      testing::dali_extra_path() + "/db/video/vfr/frames_2/"};

  LoadFrames(cfr_frames_paths, cfr_frames_);
  LoadFrames(vfr_frames_paths, vfr_frames_);
}

void VideoTestBase::LoadFrames(
  std::vector<std::string> &paths, std::vector<std::vector<cv::Mat>> &out_frames) {
  for (auto &frames_path : paths) {
    std::vector<cv::Mat> frames;
    std::ifstream frames_list(frames_path + "frames_list.txt");
    std::string frame_filename;

    while (std::getline(frames_list, frame_filename)) {
      cv::Mat frame;
      cv::cvtColor(cv::imread(frames_path + frame_filename), frame, cv::COLOR_BGR2RGB);
      DALI_ENFORCE(frame.isContinuous(), "Loaded frame is not continues in memory");

      frames.push_back(frame);
    }

    out_frames.push_back(frames);
  }
}

void VideoTestBase::CompareFrames(const uint8_t *frame, const uint8_t *gt, int size, int eps) {
  detail::parallel_for(size, [&](int start, int end, int id){ 
    for (int j = start; j < end; ++j) {
      ASSERT_NEAR(frame[j], gt[j], eps);
  }});
}

void VideoTestBase::CompareFramesAvg(const uint8_t *frame, const uint8_t *gt, int size, double eps) {
  std::vector<double> sums(std::thread::hardware_concurrency(), 0.0);

  detail::parallel_for(size, [&](int start, int end, int id){ 
    double sum = 0.0;
    for (int j = start; j < end; ++j) {
      sum += std::abs(frame[j]-gt[j]);
    }
    sums[id] = sum;
  });

  double sum = std::accumulate(sums.begin(), sums.end(), 0.0);
  sum /= size;

  // std::cout << sum << std::endl;
  
  ASSERT_LT(sum, eps);
}

void VideoTestBase::SaveFrame(uint8_t *frame, int frame_id, int sample_id, int batch_id, std::string subfolder, int width, int height, int channels) {

    TensorView<StorageCPU, uint8_t> tv(frame, TensorShape<3>{height, width, channels});
    char str[32];
    snprintf(str, 32, "/batch_%03d_sample_%03d_frame_%03d", batch_id, sample_id, frame_id);
    string path = "/home/awolant/Downloads/frames/" + subfolder + string(str) + ".png";
    testing::SaveImage(path.c_str(), tv);
}

}  // namespace dali
