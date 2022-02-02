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

#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"

#include <cuda.h>
#include <unistd.h>

#include <string>
#include <memory>

#include "dali/core/error_handling.h"
#include "dali/core/cuda_utils.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/reader/loader/video/nvdecode/color_space.h"

namespace dali {
namespace detail {
int process_video_sequence(void *user_data, CUVIDEOFORMAT *video_format) {
  return video_format->min_num_decode_surfaces;
}

int process_picture_decode(void *user_data, CUVIDPICPARAMS *picture_params) {
  FramesDecoderGpu *frames_decoder = static_cast<FramesDecoderGpu*>(user_data);

  // Sending empty packet will call this callback.
  // If we want to flush the decoder, we do not need to do anything here
  if (frames_decoder->flush_) {
    return 0;
  }

  CUDA_CALL(cuvidDecodePicture(frames_decoder->nvdecode_state_->decoder_, picture_params));

  // Process decoded frame for output
  CUVIDPROCPARAMS videoProcessingParameters = {};
  videoProcessingParameters.progressive_frame = !picture_params->field_pic_flag;
  videoProcessingParameters.second_field = 1;
  videoProcessingParameters.top_field_first = picture_params->bottom_field_flag ^ 1;
  videoProcessingParameters.unpaired_field = 0;
  videoProcessingParameters.output_stream = frames_decoder->stream_;

  uint8_t *frame_output = nullptr;

  // Take pts of the currently decoded frame
  int current_pts = frames_decoder->piped_pts_.front();
  frames_decoder->piped_pts_.pop();

  if (current_pts == frames_decoder->CurrentFramePts()) {
    // Currently decoded frame is actually the one we wanted
    frames_decoder->frame_returned_ = true;
    frame_output = frames_decoder->current_frame_output_;
  } else {
    // Put currently decoded frame to the buffer for later
    auto &slot = frames_decoder->FindEmptySlot();
    slot.pts_ = current_pts;
    frame_output = slot.frame_.data();
  }

  if (frames_decoder->current_copy_to_output_ == false) {
    return 1;
  }

  CUdeviceptr frame = {};
  unsigned int pitch = 0;

  CUDA_CALL(cuvidMapVideoFrame(
    frames_decoder->nvdecode_state_->decoder_,
    picture_params->CurrPicIdx,
    &frame,
    &pitch,
    &videoProcessingParameters));

  // TODO(awolant): Benchmark, if copy would be faster
  yuv_to_rgb(
    reinterpret_cast<uint8_t *>(frame),
    pitch,
    frame_output,
    frames_decoder->Width()* 3,
    frames_decoder->Width(),
    frames_decoder->Height(),
    frames_decoder->stream_);
  CUDA_CALL(cuvidUnmapVideoFrame(frames_decoder->nvdecode_state_->decoder_, frame));

  return 1;
}
}  // namespace detail

FramesDecoderGpu::FramesDecoderGpu(const std::string &filename, cudaStream_t stream) :
    FramesDecoder(filename),
    frame_buffer_(num_decode_surfaces_),
    stream_(stream) {
    nvdecode_state_ = std::make_unique<NvDecodeState>();

    const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
    DALI_ENFORCE(av_bsf_alloc(bsf, &bsfc_) >= 0);
    DALI_ENFORCE(avcodec_parameters_copy(
      bsfc_->par_in, av_state_->ctx_->streams[0]->codecpar) >= 0);
    DALI_ENFORCE(av_bsf_init(bsfc_) >= 0);

    filtered_packet_ = av_packet_alloc();
    DALI_ENFORCE(filtered_packet_, "Could not allocate av packet");

    // Create nv decoder
    CUVIDDECODECREATEINFO decoder_info = {};
    memset(&decoder_info, 0, sizeof(CUVIDDECODECREATEINFO));

    decoder_info.bitDepthMinus8 = 0;
    decoder_info.ChromaFormat = cudaVideoChromaFormat_420;
    decoder_info.CodecType = cudaVideoCodec_H264;
    decoder_info.ulHeight = Height();
    decoder_info.ulWidth = Width();
    decoder_info.ulMaxHeight = Height();
    decoder_info.ulMaxWidth = Width();
    decoder_info.ulTargetHeight = Height();
    decoder_info.ulTargetWidth = Width();
    decoder_info.ulNumDecodeSurfaces = num_decode_surfaces_;
    decoder_info.ulNumOutputSurfaces = 2;

    CUDA_CALL(cuvidCreateDecoder(&nvdecode_state_->decoder_, &decoder_info));

    // Create nv parser
    CUVIDPARSERPARAMS parser_info;
    memset(&parser_info, 0, sizeof(CUVIDPARSERPARAMS));
    parser_info.CodecType = cudaVideoCodec_H264;
    parser_info.ulMaxNumDecodeSurfaces = num_decode_surfaces_;
    parser_info.ulMaxDisplayDelay = 0;
    parser_info.pUserData = this;
    parser_info.pfnSequenceCallback = detail::process_video_sequence;
    parser_info.pfnDecodePicture = detail::process_picture_decode;
    parser_info.pfnDisplayPicture = nullptr;

    CUDA_CALL(cuvidCreateVideoParser(&nvdecode_state_->parser_, &parser_info));

    // Init internal frame buffer
    for (size_t i = 0; i < frame_buffer_.size(); ++i) {
      frame_buffer_[i].frame_.resize(FrameSize());
      frame_buffer_[i].pts_ = -1;
    }
}

void FramesDecoderGpu::SeekFrame(int frame_id) {
  SendLastPacket(true);
  FramesDecoder::SeekFrame(frame_id);
}

bool FramesDecoderGpu::ReadNextFrame(uint8_t *data, bool copy_to_output) {
  // No more frames in the file
  if (current_frame_ == -1) {
    return false;
  }

  // Check if requested frame was buffered earlier
  for (auto &frame : frame_buffer_) {
    if (frame.pts_ == index_[current_frame_].pts) {
      if (copy_to_output) {
        copyD2D(data, frame.frame_.data(), FrameSize());
      }
      frame.pts_ = -1;

      ++current_frame_;
      return true;
    }
  }

  current_copy_to_output_ = copy_to_output;
  current_frame_output_ = data;

  while (av_read_frame(av_state_->ctx_, av_state_->packet_) >= 0) {
    if (av_state_->packet_->stream_index != av_state_->stream_id_) {
      continue;
    }

    // Store pts from current packet to indicate,
    // that this frame is in the decoder
    piped_pts_.push(av_state_->packet_->pts);

    // Add header needed for NVDECODE to the packet
    if (filtered_packet_->data) {
      av_packet_unref(filtered_packet_);
    }
    DALI_ENFORCE(av_bsf_send_packet(bsfc_, av_state_->packet_) >= 0);
    DALI_ENFORCE(av_bsf_receive_packet(bsfc_, filtered_packet_) >= 0);

    // Prepare nv packet
    CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
    memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
    packet->payload = filtered_packet_->data;
    packet->payload_size = filtered_packet_->size;
    packet->flags = CUVID_PKT_TIMESTAMP;
    packet->timestamp = filtered_packet_->pts;

    // Send packet to the nv deocder
    frame_returned_ = false;
    CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser_, packet));

    if (frame_returned_) {
      ++current_frame_;
      return true;
    }
  }

  if (!last_frame_read_) {
    SendLastPacket();
    current_frame_ = -1;
    return true;
  }
  return false;
}

void FramesDecoderGpu::SendLastPacket(bool flush) {
  flush_ = flush;
  CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
  memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
  packet->payload = nullptr;
  packet->payload_size = 0;
  packet->flags = CUVID_PKT_ENDOFSTREAM;
  CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser_, packet));
  flush_ = false;

  if (flush) {
    last_frame_read_ = false;

    // Clear frames buffer
    for (size_t i = 0; i < frame_buffer_.size(); ++i) {
      frame_buffer_[i].pts_ = -1;
    }

    // Clear piped pts
    while (piped_pts_.size() > 0) {
      piped_pts_.pop();
    }
  }
}

BufferedFrame& FramesDecoderGpu::FindEmptySlot() {
  for (auto &frame : frame_buffer_) {
    if (frame.pts_ == -1) {
      return frame;
    }
  }
  DALI_FAIL("Could not find empty slot in the frame buffer");
}


void FramesDecoderGpu::Reset() {
  SendLastPacket(true);
  FramesDecoder::Reset();
}
}  // namespace dali
