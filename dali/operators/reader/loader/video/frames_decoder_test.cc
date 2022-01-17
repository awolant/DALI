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

#include <exception>
#include <cuda_runtime_api.h>

#include "dali/test/dali_test_config.h"
#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/video/video_test_base.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"


namespace dali {
class FramesDecoderTest : public VideoTestBase {
};


TEST_F(FramesDecoderTest, ConstantFrameRate) {
    std::string path = testing::dali_extra_path() + "/db/video/cfr/test_1.mp4";

    // Create file, build index
    FramesDecoder file(path);

    ASSERT_EQ(file.Height(), 720);
    ASSERT_EQ(file.Width(), 1280);
    ASSERT_EQ(file.Channels(), 3);
    ASSERT_EQ(file.NumFrames(), 50);

    std::vector<uint8_t> frame(file.FrameSize());

    // Read first frame
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 0), file.FrameSize());

    // Seek to frame
    file.SeekFrame(25);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 25), file.FrameSize());

    // Seek back to frame
    file.SeekFrame(12);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 12), file.FrameSize());

    // Seek to last frame (flush frame)
    file.SeekFrame(49);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 49), file.FrameSize());

    // Wrap around to first frame
    ASSERT_FALSE(file.ReadNextFrame(frame.data()));
    file.Reset();
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 0), file.FrameSize());
}

TEST_F(FramesDecoderTest, VariableFrameRate) {
    std::string path = testing::dali_extra_path() + "/db/video/vfr/test_2.mp4";

    // Create file, build index
    FramesDecoder file(path);

    ASSERT_EQ(file.Height(), 600);
    ASSERT_EQ(file.Width(), 800);
    ASSERT_EQ(file.Channels(), 3);
    ASSERT_EQ(file.NumFrames(), 60);

    std::vector<uint8_t> frame(file.FrameSize());

    // Read first frame
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());

    // Seek to frame
    file.SeekFrame(25);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 25), file.FrameSize());

    // Seek back to frame
    file.SeekFrame(12);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 12), file.FrameSize());

    // Seek to last frame (flush frame)
    file.SeekFrame(59);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 59), file.FrameSize());

    // Wrap around to first frame
    ASSERT_FALSE(file.ReadNextFrame(frame.data()));
    file.Reset();
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());
}

// TEST_F(FramesDecoderTest, VariableFrameRateGpu) {
//     CUresult cuResult = cuInit(0);
//     CUdevice device = { 0 };
//     cuResult = cuDeviceGet(&device, 0);
//     CUcontext context = { 0 };
//     cuResult = cuCtxCreate(&context, 0, device);


//     std::string path = testing::dali_extra_path() + "/db/video/vfr/test_2.avi";
//     // std::string path = testing::dali_extra_path() + "/db/video/vfr/test_2.mp4";

//     // Create file, build index
//     FramesDecoderGpu file(path);
//     // FramesDecoder file(path);

//     ASSERT_EQ(file.Height(), 600);
//     ASSERT_EQ(file.Width(), 800);
//     ASSERT_EQ(file.Channels(), 3);
//     ASSERT_EQ(file.NumFrames(), 60);

//     uint8_t *frame = nullptr;
//     cuMemAlloc(
//         (CUdeviceptr*)&frame,
//         file.FrameSize());
//     std::vector<uint8_t> frame_cpu(file.FrameSize());

//     // Read first frame
//     file.ReadNextFrame(frame);
//     cudaMemcpy(
//         frame_cpu.data(),
//         frame,
//         file.FrameSize() * sizeof(uint8_t),
//         cudaMemcpyDeviceToHost);
//     this->CompareFramesAvg(frame_cpu.data(), this->GetVfrFrame(1, 0), file.FrameSize());
//     this->SaveFrame(frame_cpu.data(), 0, 0, 0, "reader", 800, 600, 3);
//     this->SaveFrame(this->GetVfrFrame(1, 0), 0, 0, 0, "gt", 800, 600, 3);

//     // // Seek to frame
//     file.SeekFrame(25);
//     file.ReadNextFrame(frame);
//     cudaMemcpy(
//         frame_cpu.data(),
//         frame,
//         file.FrameSize() * sizeof(uint8_t),
//         cudaMemcpyDeviceToHost);
//     this->CompareFramesAvg(frame_cpu.data(), this->GetVfrFrame(1, 25), file.FrameSize());
//     this->SaveFrame(frame_cpu.data(), 25, 0, 0, "reader", 800, 600, 3);
//     this->SaveFrame(this->GetVfrFrame(1, 25), 25, 0, 0, "gt", 800, 600, 3);

//     // Seek back to frame
//     file.SeekFrame(12);
//     file.ReadNextFrame(frame);
//     cudaMemcpy(
//         frame_cpu.data(),
//         frame,
//         file.FrameSize() * sizeof(uint8_t),
//         cudaMemcpyDeviceToHost);
//     this->CompareFramesAvg(frame_cpu.data(), this->GetVfrFrame(1, 12), file.FrameSize());
//     this->SaveFrame(frame_cpu.data(), 12, 0, 0, "reader", 800, 600, 3);
//     this->SaveFrame(this->GetVfrFrame(1, 12), 12, 0, 0, "gt", 800, 600, 3);

//     // Seek to last frame (flush frame)
//     file.SeekFrame(59);
//     file.ReadNextFrame(frame);
//     cudaMemcpy(
//         frame_cpu.data(),
//         frame,
//         file.FrameSize() * sizeof(uint8_t),
//         cudaMemcpyDeviceToHost);
//     this->CompareFramesAvg(frame_cpu.data(), this->GetVfrFrame(1, 59), file.FrameSize());
//     this->SaveFrame(frame_cpu.data(), 59, 0, 0, "reader", 800, 600, 3);
//     this->SaveFrame(this->GetVfrFrame(1, 59), 59, 0, 0, "gt", 800, 600, 3);

//     // // Wrap around to first frame
//     // ASSERT_FALSE(file.ReadNextFrame(frame.data()));
//     // file.Reset();
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFrames(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());

//     // std::string path = testing::dali_extra_path() + "/db/video/vfr/test_2.mp4";

//     // // Create file, build index
//     // FramesDecoder file(path);

//     // ASSERT_EQ(file.Height(), 600);
//     // ASSERT_EQ(file.Width(), 800);
//     // ASSERT_EQ(file.Channels(), 3);
//     // ASSERT_EQ(file.NumFrames(), 60);

//     // std::vector<uint8_t> frame(file.FrameSize());

//     // // Read first frame
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());

//     // // Seek to frame
//     // file.SeekFrame(25);
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 25), file.FrameSize());

//     // // Seek back to frame
//     // file.SeekFrame(12);
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 12), file.FrameSize());

//     // // Seek to last frame (flush frame)
//     // file.SeekFrame(59);
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 59), file.FrameSize());

//     // file.ReadNextFrame(frame.data());

//     // // Wrap around to first frame
//     // ASSERT_FALSE(file.ReadNextFrame(frame.data()));
//     // file.Reset();
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());
// }

// TEST_F(FramesDecoderTest, VariableFrameRateAvi) {
//     std::string path = testing::dali_extra_path() + "/db/video/vfr/test_2.avi";

//     // Create file, build index
//     FramesDecoder file(path);

//     ASSERT_EQ(file.Height(), 600);
//     ASSERT_EQ(file.Width(), 800);
//     ASSERT_EQ(file.Channels(), 3);
//     ASSERT_EQ(file.NumFrames(), 60);

//     std::vector<uint8_t> frame(file.FrameSize());

//     // Read first frame
//     file.ReadNextFrame(frame.data());
//     this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());

//     // Seek to frame
//     file.SeekFrame(25);
//     file.ReadNextFrame(frame.data());
//     this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 25), file.FrameSize());

//     // Seek back to frame
//     file.SeekFrame(12);
//     file.ReadNextFrame(frame.data());
//     this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 12), file.FrameSize());

//     // Seek to last frame (flush frame)
//     file.SeekFrame(59);
//     file.ReadNextFrame(frame.data());
//     this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 59), file.FrameSize());

//     // Wrap around to first frame
//     ASSERT_FALSE(file.ReadNextFrame(frame.data()));
//     file.Reset();
//     file.ReadNextFrame(frame.data());
//     this->CompareFramesAvg(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());
// }

TEST_F(FramesDecoderTest, InvalidPath) {
    std::string path = "invalid_path.mp4";

    try {
        FramesDecoder file(path);
    } catch (const DALIException &e) {
        EXPECT_TRUE(strstr(
            e.what(),
            make_string("Failed to open video file at path ", path).c_str()));
    }
}

TEST_F(FramesDecoderTest, NoVideoStream) {
    std::string path = testing::dali_extra_path() + "/db/audio/wav/dziendobry.wav";

    try {
        FramesDecoder file(path);
    } catch (const DALIException &e) {
        EXPECT_TRUE(strstr(
            e.what(),
            make_string("Could not find a valid video stream in a file ", path).c_str()));
    }
}

TEST_F(FramesDecoderTest, InvalidSeek) {
    std::string path = testing::dali_extra_path() + "/db/video/cfr/test_1.mp4";
    FramesDecoder file(path);

    try {
        file.SeekFrame(60);
    } catch (const DALIException &e) {
        EXPECT_TRUE(strstr(
            e.what(),
            "Invalid seek frame id. frame_id = 60, num_frames = 50"));
    }
}

}  // namespace dali
