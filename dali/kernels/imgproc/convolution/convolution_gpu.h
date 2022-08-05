// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_GPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_GPU_H_

#include "dali/core/boundary.h"
#include "dali/core/convert.h"
#include "dali/core/format.h"
#include "dali/core/span.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/cutlass/device/gemm.h"
#include "dali/kernels/imgproc/convolution/cutlass/utility.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/reduce/online_reducer.h"

namespace dali {
namespace kernels {

struct ConvEpilogue {
  ConvEpilogue(span<const float> scales, float beta = 0.f)  // NOLINT
      : scales_{scales}, scale_{1.f}, beta_{beta} {}
  ConvEpilogue(float scale, float beta = 0.f) : scales_{}, scale_{scale}, beta_{beta} {}  // NOLINT

  inline int num_samples() const {
    return scales_.size();
  }

  inline float alpha(int sample_idx) const {
    return scales_.empty() ? scale_ : scales_[sample_idx];
  }

  inline float beta() const {
    return beta_;
  }

 private:
  span<const float> scales_;
  float scale_;
  float beta_;
};

/**
 * @brief Apply a convolution with a 1-channel `window` in the specified axis.
 *
 * If the anchor is not provided, the window is centered over input:
 * `window_anchor = window_size / 2`
 *
 * Only odd windows with centered anchor are now supported.
 *
 * Operation is done by using GEMM (General Matrix Multiplication) and generating the matrix
 * from convolution window on the fly.
 *
 * CUTLASS allows to calculate GEMM: ``D = alpha * A * B + beta * C``.
 * C and D can be the same matrix, giving us an equivalent of: ``D *= beta; D += alpha * A * B``
 *
 * To calculate the convolution, we modified the GEMM implementation so that it generates one of the
 * input matrices on the fly based on the convolution window.
 * Such a matrix can be generated by applying the convolution window to an identity matrix of the
 * desired size.
 *
 * For the convolution we pass D = C = `out`.
 * Parameters `alpha` and `beta` can be customized by passing `conv_epilogue` to `Run` method.
 * By default, a single `alpha` is used for the entire batch but it can be specified
 * per sample as a span of floats. The `beta` parameter defaults to 0, so that
 * the C matrix is zeroed.
 * A and B correspond to `in` and `window` (the order depends on whether it's an inner or an outer
 * convolution). Effectively, we calculate ``D = alpha * A * B + beta * D``.
 */
template <typename Out, typename In, typename W, int ndim, int axis, bool has_channels = true>
struct ConvolutionGpu {
  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
                           const TensorListShape<1>& window_size) {
    KernelRequirements req;
    ScratchpadEstimator se;
    DALI_ENFORCE(
        in_shape.size() == window_size.size(),
        make_string(
            "Provided input shape and window sizes should have the same number of samples. Got: ",
            in_shape.size(), " vs ", window_size.size(), "."));
    int num_samples = in_shape.size();
    for (int i = 0; i < num_samples; i++) {
      int num_channels = has_channels ? in_shape[i][ndim - 1] : 1;
      DALI_ENFORCE(
          window_size[i][0] % 2 == 1,
          make_string(
              "Even or non-centered windows are not supported yet, got window with even length: ",
              window_size, " for sample ", i, "."));

      DALI_ENFORCE(window_size[i][0] * num_channels < kMaxWindowSize,
                   make_string("Window is too big for sample ", i, ", got: ", window_size[i][0],
                               ", expected at most: ", kMaxWindowSize / num_channels, "."));
    }
    se.add<mm::memory_kind::host, W>(num_samples * kWindowCopyBufferSize);
    se.add<mm::memory_kind::device, W>(num_samples * kWindowCopyBufferSize);
    se.add<mm::memory_kind::device, typename CutlassConv::SampleParams>(num_samples);
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(in_shape);
    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim> out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const TensorListView<StorageCPU, const W, 1>& windows,
           const span<const int> window_anchors = {}, const ConvEpilogue& conv_epilogue = 1.f) {
    int num_samples = in.size();
    int num_scales = conv_epilogue.num_samples();

    DALI_ENFORCE(
        window_anchors.size() == num_samples || window_anchors.size() == 0,
        make_string(
            "Unexpected number of window_anchors, expected either anchors for all samples ( ",
            num_samples,
            ") or no anchors for windows centered by default, got: ", window_anchors.size(), "."));

    DALI_ENFORCE(
        num_scales == 0 || num_scales == num_samples,
        make_string(
            "Scale argument must be either a scalar or a span of length equal to the batch size (",
            num_samples, "), got: ", num_scales, "."));

    auto* window_tmp_buffer_host_ptr =
        ctx.scratchpad->AllocateHost<W>(num_samples * kWindowCopyBufferSize);
    span<W> window_tmp_buffer_host(window_tmp_buffer_host_ptr, num_samples * kWindowCopyBufferSize);

    // Pad and align windows in tmp memory, transfer the aligned windows to GPU
    FillAlignedWindows(window_tmp_buffer_host, windows, in.shape);
    // To*GPU allocates num_samples * kWindowCopyBufferSize * sizeof(W) memory through scratchpad
    // Use ToContiguousGPU to use DALI's pinned buffer instead of system one.
    auto* window_tmp_buffer_gpu =
        std::get<0>(ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, window_tmp_buffer_host));

    Arguments args;
    args.device_params_ptr =
        ctx.scratchpad->AllocateGPU<typename CutlassConv::SampleParams>(num_samples);

    if (kIsInnerConv) {
      // Inner (innermost) - repack arguments
      for (int i = 0; i < num_samples; i++) {
        int window_size = static_cast<int>(windows.tensor_shape_span(i)[0]);
        // convert anchor to the correlation formula used internally
        int window_anchor =
            window_size - 1 - (window_anchors.size() ? window_anchors[i] : window_size / 2);
        DALI_ENFORCE(
            window_anchor == window_size / 2,
            make_string("Support for non-centered window is not yet implemented, got anchor: ",
                        window_anchor, ", expected: ", window_size / 2, "."));
        cutlass::Array<int, 2> size;
        auto sample_shape = in.tensor_shape(i);
        int num_channels = has_channels ? sample_shape[ndim - 1] : 1;
        // height
        size[0] = volume(sample_shape.begin(), sample_shape.begin() + axis);
        // width
        size[1] = sample_shape[axis];
        // Special case for extent equal 1, see FillAlignedWindows for details
        if (size[1] == 1) {
          window_size = 1;
          window_anchor = 0;
        }
        int row_stride = sample_shape[axis] * num_channels;
        auto* window_gpu =
            reinterpret_cast<cutlass_W*>(window_tmp_buffer_gpu + i * kWindowCopyBufferSize);
        const auto* cutlass_in = reinterpret_cast<const cutlass_In*>(in.tensor_data(i));
        auto* cutlass_out = reinterpret_cast<cutlass_Out*>(out.tensor_data(i));
        float alpha = conv_epilogue.alpha(i);
        float beta = conv_epilogue.beta();
        args.sample_arguments.push_back(SampleArguments{
            size,                       // Input matrix dimensions
            window_size,                // Window size
            window_anchor,              // Window anchor
            num_channels,               // channels count (innermost)
            {cutlass_in, row_stride},   // Tensor-ref for source matrix A
            window_gpu,                 // Pointers to windows
            {cutlass_out, row_stride},  // Tensor-ref for source matrix C
            {cutlass_out, row_stride},  // Tensor-ref for destination matrix D
            {alpha, beta}               // Epilogue scalars
        });
      }
    } else {
      // Outer or not-innermost - repack arguments
      for (int i = 0; i < num_samples; i++) {
        int window_size = static_cast<int>(windows.tensor_shape_span(i)[0]);
        // convert anchor to the correlation formula used internally
        int window_anchor =
            window_size - 1 - (window_anchors.size() ? window_anchors[i] : window_size / 2);
        DALI_ENFORCE(
            window_anchor == window_size / 2,
            make_string("Support for non-centered window is not yet implemented, got anchor: ",
                        window_anchor, ", expected: ", window_size / 2, "."));
        cutlass::Array<int, 2> size;
        auto sample_shape = in.tensor_shape(i);
        // height
        size[0] = sample_shape[axis];
        // width
        size[1] = volume(sample_shape.begin() + axis + 1, sample_shape.end());
        // Special case for extent equal 1, see FillAlignedWindows for details
        if (size[0] == 1) {
          window_size = 1;
          window_anchor = 0;
        }
        auto strides = GetStrides(sample_shape);
        int row_stride = strides[axis];
        int planes = volume(sample_shape.begin(), sample_shape.begin() + axis);
        int plane_stride = axis > 0 ? strides[axis - 1] : 0;
        auto* window_gpu =
            reinterpret_cast<cutlass_W*>(window_tmp_buffer_gpu + i * kWindowCopyBufferSize);
        const auto* cutlass_in = reinterpret_cast<const cutlass_In*>(in.tensor_data(i));
        auto* cutlass_out = reinterpret_cast<cutlass_Out*>(out.tensor_data(i));
        float alpha = conv_epilogue.alpha(i);
        float beta = conv_epilogue.beta();
        args.sample_arguments.push_back(
            SampleArguments{size,                       // Input matrix dimensions
                            window_size,                // Window size
                            window_anchor,              // Window anchor
                            1,                          // channels don't matter for outer dim
                            {cutlass_in, row_stride},   // Tensor-ref for source matrix A
                            window_gpu,                 // Pointers to windows
                            {cutlass_out, row_stride},  // Tensor-ref for source matrix C
                            {cutlass_out, row_stride},  // Tensor-ref for destination matrix D
                            {alpha, beta},              // Epilogue scalars
                            planes,                     // For non-outermost we can have 1+ planes
                            plane_stride});
      }
    }
    // Construct and invoke the CUTLASS kernel
    CutlassConv gemm_operator;
    auto status = gemm_operator.can_implement(args);
    DALI_ENFORCE(status == cutlass::Status::kSuccess,
                 make_string("Operation not possible: ", cutlass::cutlassGetStatusString(status)));
    gemm_operator(args, ctx.gpu.stream);
  }

 private:
  // Innermost convolution requires channel handling and multiplies by "kernel matrix"
  // (matrix generated based on convolution kernel windows) from right.
  // Non-innermost convolutions are channel agnostic (assume channels = 1) and place the
  // generated matrix on the left hand side of GEMM.
  static constexpr bool kIsInnerConv = axis == ndim - has_channels - 1;
  static constexpr int kLastSpatialDim = ndim - has_channels - 1;

  using RowMajor = cutlass::layout::RowMajor;

  using CutlassWindowConfig = cutlass::gemm::ConvWindowConfiguration<1024, kIsInnerConv>;

  using cutlass_In = cutlass::to_cutlass_t<In>;
  using cutlass_W = cutlass::to_cutlass_t<W>;
  using cutlass_Out = cutlass::to_cutlass_t<Out>;

  // Basic SIMT kernel with no additional conversions
  // 2nd and 5th template parameter (In and W now) allow for additional conversion when loading
  // data inside the cutlass kernel. For now it's no-op (hence the same types twice).
  // TODO(klecki): Consider adjusting `ElementAccumulator` from float to some integral type
  // when calculating convolution of integral input and integral kernel.
  using CutlassConv = typename cutlass::gemm::device::Conv<
      cutlass_In,           /// Data-type of Input matrix
      cutlass_In,           /// Additional cast for Input matrix type when loading
      RowMajor,             /// Layout of Input matrix
      cutlass_W,            /// Data-type of Conv window
      cutlass_W,            /// Additional cast for Conv window type when loading
      cutlass_Out,          /// Data-type of Output matrix
      RowMajor,             /// Layout of Output matrix
      kIsInnerConv,         /// convolution kind
      CutlassWindowConfig,  /// Size and layout of SMEM for window kernel lookups
      float>;               /// Element type for internal accumulation


  static constexpr int kMaxWindowSize = CutlassConv::ConvWindowConfiguration::kMaxWindowSize;
  static constexpr int kWindowCopyBufferSize =
      CutlassConv::ConvWindowConfiguration::kTotalAlignedSize;

  using Arguments = typename CutlassConv::Arguments;

  using SampleArguments = typename CutlassConv::SampleArguments;

  static_assert(0 <= axis && axis <= kLastSpatialDim,
                "Selected axis must be in [0, ndim) when there is no channel axis, or in [0, ndim "
                "- 1) for channel-last input");


  void FillAlignedWindow(span<W> window_tmp_buffer_host, int window_idx, span<const W> window_src,
                         const TensorListShape<ndim>& in_shape) {
    using dst_win_t = typename CutlassConv::ConvWindowConfiguration::template PaddedWindowBuffer<W>;
    int num_channels = has_channels ? in_shape[window_idx][ndim - 1] : 1;
    auto window_padded_dst = dst_win_t(&window_tmp_buffer_host[window_idx * kWindowCopyBufferSize]);
    CutlassConv::ConvWindowConfiguration::prepare_window(window_padded_dst, window_src,
                                                         num_channels);
  }

  /**
   * @brief Repack kernel windows from packed TLV representation to a padded layout
   * suitablbe for CUTLASS kernel taking into account the shape.
   *
   * For axis of extent 1 the kernel window is compacted to 1 element.
   */
  void FillAlignedWindows(span<W> window_tmp_buffer_host,
                          const TensorListView<StorageCPU, const W, 1>& window,
                          const TensorListShape<ndim>& in_shape) {
    for (int i = 0; i < window.num_samples(); i++) {
      // Special case, for axis with extent 1 - we need to sum the window to 1 element,
      // to make it work with border reflect 101. For the CPU version, the reflect 101
      // works as border repeat - so with 1 elements it works by calculating
      // (sum(window) * input_element). The kernel matrix generation ends as infinite loop
      // for input of extent equal 1, so we compact the kernel to 1 element (what would effectively
      // happen either way) and just lookup that one elemnt in the CUTLASS kernels
      if (in_shape[i][axis] == 1) {
        OnlineReducer<W, reductions::sum> r;
        r.reset();
        for (int win_elem = 0; win_elem < window.tensor_shape_span(i)[0]; win_elem++) {
          r.add(window.tensor_data(i)[win_elem]);
        }
        std::array<W, 1> tmp_window = {r.result()};
        auto window_src = make_span(tmp_window.data(), 1);
        FillAlignedWindow(window_tmp_buffer_host, i, window_src, in_shape);
      } else {
        auto window_src = make_span(window.tensor_data(i), window.tensor_shape_span(i)[0]);
        FillAlignedWindow(window_tmp_buffer_host, i, window_src, in_shape);
      }
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_GPU_H_
