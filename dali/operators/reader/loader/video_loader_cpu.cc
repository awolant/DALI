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

extern "C" {
#include <libswscale/swscale.h>
}

#include "dali/operators/reader/loader/video_loader_cpu.h"

namespace dali {

VideoFileCPU::VideoFileCPU(std::string &filename) {
  ctx_ = avformat_alloc_context();
  avformat_open_input(&ctx_, filename.c_str(), nullptr, nullptr);

  for (size_t i = 0; i < ctx_->nb_streams; ++i) {
    auto stream = ctx_->streams[i];
    codec_params_ = ctx_->streams[i]->codecpar;
    codec_ = avcodec_find_decoder(codec_params_->codec_id);

    if (codec_->type == AVMEDIA_TYPE_VIDEO) {
        stream_id_ = i;
        break;
    }
  }

  codec_ctx_ = avcodec_alloc_context3(codec_);
  avcodec_parameters_to_context(codec_ctx_, codec_params_);
  avcodec_open2(codec_ctx_, codec_, nullptr);
}


void VideoLoaderCPU::ReadSample(Tensor<CPUBackend> &sample) {
  auto &sample_span = sample_spans_[current_index_];
  auto &video_file = video_files_[0];

  ++current_index_;
  MoveToNextShard(current_index_);

  sample.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
  sample.Resize(
    TensorShape<4>{sequence_len_, video_file.width(), video_file.height(), video_file.channels_});

  //TODO: przesunąć na kolejną ramkę z sekwencji (sekwencja jest dłuższa niż jedna ramka)
  auto data = sample.mutable_data<uint8_t>();

  AVFrame *frame = av_frame_alloc();
  AVPacket *packet = av_packet_alloc();

  int ret = -1;
  int frames_count = 0;
  while ((ret = av_read_frame(video_file.ctx_, packet)) >= 0) {
    if (packet->stream_index != video_file.stream_id_) {
      continue;
    }

    ret = avcodec_send_packet(video_file.codec_ctx_, packet);
    ret = avcodec_receive_frame(video_file.codec_ctx_, frame);

    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      continue;
    }

    

    SwsContext  *sws_ctx = sws_getContext(
      frame->width, 
      frame->height, 
      video_file.codec_ctx_->pix_fmt, 
      frame->width, 
      frame->height, 
      AV_PIX_FMT_RGB24, 
      SWS_BILINEAR, 
      nullptr, 
      nullptr, 
      nullptr);

    uint8_t *dest[4] = {data, nullptr, nullptr, nullptr};
    int dest_linesize[4] = {frame->width * 3, 0, 0, 0};
    sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, dest, dest_linesize);

    ++frames_count;
    
    if (frames_count == sequence_len_) {
      break;
    }
  }

  if (frames_count == sequence_len_) {
    return;
  }

  ret = avcodec_send_packet(video_file.codec_ctx_, nullptr);

  ret = avcodec_receive_frame(video_file.codec_ctx_, frame);

  if (ret < 0) {
    return;
  }

  SwsContext  *sws_ctx = sws_getContext(
    frame->width, 
    frame->height, 
    video_file.codec_ctx_->pix_fmt, 
    frame->width, 
    frame->height, 
    AV_PIX_FMT_RGB24, 
    SWS_BILINEAR, 
    nullptr, 
    nullptr, 
    nullptr);

  uint8_t *dest[4] = {data, nullptr, nullptr, nullptr};
  int dest_linesize[4] = {frame->width * 3, 0, 0, 0};
  sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, dest, dest_linesize);

  ++frames_count;

  if (frames_count == sequence_len_) {
    return;
  }
}

Index VideoLoaderCPU::SizeImpl() {
  return sample_spans_.size();
}

void VideoLoaderCPU::PrepareMetadataImpl() {
  for (auto &filename : filenames_) {
    video_files_.push_back(VideoFileCPU(filename));
  }

  for (int start = 0; start + sequence_len_ <= video_files_[0].nb_frames(); start += sequence_len_) {
    sample_spans_.push_back(VideoSampleSpan(start, start + sequence_len_));
  }
}

void VideoLoaderCPU::Reset(bool wrap_to_shard) {
  current_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, SizeImpl()) : 0;
}

}  // namespace dali
