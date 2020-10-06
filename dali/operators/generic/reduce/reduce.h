// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_REDUCE_H_
#define DALI_OPERATORS_GENERIC_REDUCE_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {

class Reduce : public Operator<CPUBackend> {
 public:
  explicit inline Reduce(const OpSpec &spec) :
    Operator<CPUBackend>(spec) {}

    bool CanInferOutputs() const override { return true; }

  inline ~Reduce() override = default;

  DISABLE_COPY_MOVE_ASSIGN(Reduce);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override {
    output_desc.resize(1);
    auto &input = ws.template InputRef<CPUBackend>(0);

    int batch_size = input.shape().num_samples();

    output_desc[0].type =  input.type();
    output_desc[0].shape = input.shape();;
    TensorShape<1> sample_shape {1};

    for (int i = 0; i < batch_size; ++i) {
      output_desc[0].shape.set_tensor_shape(i, sample_shape);
    }

    return true;
  }

  void RunImpl(workspace_t<CPUBackend> &ws) override {
    auto& in = ws.InputRef<CPUBackend>(0);
    auto in_view = view<const int>(in);

    auto &out = ws.OutputRef<CPUBackend>(0);
    auto out_view = view<int>(out);

    for (int sample = 0; sample < in_view.num_samples(); sample++) {
      auto sample_view = in_view[sample];
      int sum = 0;
      for (int elem = 0; elem < sample_view.num_elements(); ++elem) {
        sum += sample_view.data[elem];
      }

      out_view[sample].data[0] = sum;
    }
  }

 private:

  DALIDataType output_type_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_REDUCE_H_
