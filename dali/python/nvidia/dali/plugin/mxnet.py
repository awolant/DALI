# Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvidia.dali.backend import TensorGPU, TensorListGPU
from nvidia.dali import types
from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import mxnet as mx
import ctypes
import numpy as np
from collections import Iterable


##################################################
##################################################
################## Common utils ##################
##################################################
##################################################


# MXNet currently does not expose WaitToWrite C API call
# in Python API
def _wait_to_write(arr):
    if not isinstance(arr, mx.nd.NDArray):
        raise RuntimeError("Can only wait for NDArray")
    mx.base._LIB.MXNDArrayWaitToWrite(arr.handle)

def feed_ndarray(dali_tensor, arr, cuda_stream = None):
    """
    Copy contents of DALI tensor to MXNet's NDArray.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : mxnet.nd.NDArray
            Destination of the copy
    `cuda_stream` : cudaStream_t handle or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using the default internal user stream or stream 0
                    is expected.
    """
    # Wait until arr is no longer used by the engine
    _wait_to_write(arr)
    assert dali_tensor.shape() == list(arr.shape), \
            ("Shapes do not match: DALI tensor has shape {0}"
            ", but NDArray has shape {1}".format(dali_tensor.shape(), list(arr.shape)))
    # Get CTypes void pointer to the underlying memory held by arr
    ptr = ctypes.c_void_p()
    mx.base._LIB.MXNDArrayGetData(arr.handle, ctypes.byref(ptr))

    cuda_stream = types._raw_cuda_stream(cuda_stream)

    # Copy data from DALI tensor to ptr
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        dali_tensor.copy_to_external(ptr, None if cuda_stream is None else ctypes.c_void_p(cuda_stream))
    else:
        dali_tensor.copy_to_external(ptr)

class _DALIMXNetIteratorBase(mx.io.DataIter, _DaliBaseIterator):
    """
    Base class with methods shared by both DALIGenericIterator and DALIGluonIterator.
    """
    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 fill_last_batch=None,
                 last_batch_padded=False,
                 auto_reset=False,
                 last_batch_policy=LastBatchPolicy.FILL):
        _DaliBaseIterator.__init__(self, pipelines, size, reader_name, auto_reset,
                                   fill_last_batch, last_batch_padded, last_batch_policy)

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__()

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        _DaliBaseIterator.reset(self)

def get_mx_array(shape, ctx=None, dtype=None):
    # WAR
    # ToDo (jlisiecki) - fix when upstream MXNet fixes this
    # mx.nd.empty doesn't support np.longlong as mx.nd.zeros does, but it does np.int64
    # which from our point of view is the same
    if dtype == np.longlong:
        dtype = np.int64
    # mx.nd.empy doesn't handle scalaras as shape
    if not isinstance(shape, Iterable):
        shape = [shape]

    return mx.nd.empty(shape, ctx, dtype)


###################################################
###################################################
################## MXNet Sym API ##################
###################################################
###################################################

class DALIGenericIterator(_DALIMXNetIteratorBase):
    """
    General DALI iterator for MXNet. It can return any number of
    outputs from the DALI pipeline in the form of MXNet's DataBatch
    of NDArrays.

    Please keep in mind that NDArrays returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another NDArray.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of (str, str)
                 List of pairs (output_name, tag) which maps consecutive
                 outputs of DALI pipelines to proper field in MXNet's
                 DataBatch.
                 tag is one of DALIGenericIterator.DATA_TAG
                 and DALIGenericIterator.LABEL_TAG mapping given output
                 for data or label correspondingly.
                 output_names should be distinct.
    size : int, default = -1
          Number of samples in the shard for the wrapped pipeline (if there is more than one it is a sum)
          Providing -1 means that the iterator will work until StopIteration is raised
          from the inside of iter_setup(). The options `last_batch_policy`, `last_batch_padded` and
          `auto_reset` don't work in such case. It works with only one pipeline inside
          the iterator.
          Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy` to
                PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    data_layout : str, optional, default = 'NCHW'
                  Either 'NHWC' or 'NCHW' - layout of the pipeline outputs.
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    squeeze_labels: (DEPRECATED) bool, optional, default = False
                 Whether the iterator should squeeze the labels before
                 copying them to the ndarray.
                 This argument is deprecated and will be removed from future releases.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the mxnet.ndarray will be resized accordingly
                 if the shape of DALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy : default = FILL
                What to do with the last batch when there is no enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch (it doesn't literally drop but sets ``pad`` field of ndarray
                so the following code could use it to drop the data). If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = PARTIAL, last_batch_padded = True  -> last batch = ``[7]``, next iteration will return ``[1, 2]``

    last_batch_policy = PARTIAL, last_batch_padded = False -> last batch = ``[7]``, next iteration will return ``[2, 3]``

    last_batch_policy = FILL, last_batch_padded = True   -> last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    last_batch_policy = FILL, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``

    last_batch_policy = DROP, last_batch_padded = True   -> last batch = ``[5, 6]``, next iteration will return ``[1, 2]``

    last_batch_policy = DROP, last_batch_padded = False  -> last batch = ``[5, 6]``, next iteration will return ``[2, 3]``
    """
    def __init__(self,
                 pipelines,
                 output_map,
                 size=-1,
                 reader_name=None,
                 data_layout='NCHW',
                 fill_last_batch=None,
                 auto_reset=False,
                 squeeze_labels=True,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL):

        # check the assert first as _DaliBaseIterator would run the prefetch
        self._output_names_map = [x[0] for x in output_map]
        self._output_categories_map = [x[1] for x in output_map]
        self._output_categories = {DALIGenericIterator.DATA_TAG, DALIGenericIterator.LABEL_TAG}
        assert set(self._output_categories_map) <= self._output_categories, \
            "Only DATA_TAG and LABEL_TAG are allowed"
        assert len(set(self._output_names_map)) == len(self._output_names_map), \
            "output_names in output_map should be distinct"
        self.output_map = output_map

        super(DALIGenericIterator, self).__init__(
            pipelines,
            size,
            reader_name,
            fill_last_batch,
            last_batch_padded,
            auto_reset,
            last_batch_policy)
        self._squeeze_labels = squeeze_labels
        self._dynamic_shape = dynamic_shape
        # Use double-buffering of data batches
        self._data_batches = [[None] for i in range(self._num_gpus)]
        self._current_data_batch = 0

        self._first_batch = None
        try:
            self._first_batch = DALIGenericIterator.__next__(self)
        except StopIteration:
            assert False, "It seems that there is no data in the pipeline. This may happen if `last_batch_policy` is set to PARTIAL and the requested batch size is greater than the shard size."
        # Set data descriptors for MXNet
        self.provide_data = []
        self.provide_label = []

        category_names = {key : [] for key in self._output_categories}
        for name, category in output_map:
            category_names[category].append(name)
        for i, data in enumerate(self._first_batch[0].data):
            data_shape  = (data.shape[0] * self._num_gpus,) + data.shape[1:]
            self.provide_data.append(mx.io.DataDesc(category_names[DALIGenericIterator.DATA_TAG][i], \
                data_shape, data.dtype, layout=data_layout))
        for i, label in enumerate(self._first_batch[0].label):
            label_shape = (label.shape[0] * self._num_gpus,) + label.shape[1:]
            self.provide_label.append(mx.io.DataDesc(category_names[DALIGenericIterator.LABEL_TAG][i], \
                label_shape, label.dtype))

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        outputs = self._get_outputs()

        for i in range(self._num_gpus):
            # MXNet wants batches with clear distinction between
            # data and label entries, so segregate outputs into
            # 2 categories
            category_outputs = {key : [] for key in self._output_categories}
            for j, out in enumerate(outputs[i]):
                category_outputs[self._output_categories_map[j]].append(out)
            # Change DALI TensorLists into Tensors
            category_tensors = dict()
            category_info = dict()
            # For data proceed normally
            category_tensors[DALIGenericIterator.DATA_TAG] = \
                [x.as_tensor() for x in category_outputs[DALIGenericIterator.DATA_TAG]]
            category_info[DALIGenericIterator.DATA_TAG] = \
                [(x.shape(), np.dtype(x.dtype())) for x in category_tensors[DALIGenericIterator.DATA_TAG]]
            # For labels we squeeze the tensors
            category_tensors[DALIGenericIterator.LABEL_TAG] = \
                [x.as_tensor() for x in category_outputs[DALIGenericIterator.LABEL_TAG]]
            if self._squeeze_labels:
                for label in category_tensors[DALIGenericIterator.LABEL_TAG]:
                    label.squeeze(-1)  # Squeeze last dimension if necessary
            category_info[DALIGenericIterator.LABEL_TAG] = \
                [(x.shape(), np.dtype(x.dtype())) for x in category_tensors[DALIGenericIterator.LABEL_TAG]]

            # If we did not yet allocate memory for that batch, do it now
            if self._data_batches[i][self._current_data_batch] is None:
                mx_gpu_device = mx.gpu(self._pipes[i].device_id)
                mx_cpu_device = mx.cpu(0)
                category_device = {key : [] for key in self._output_categories}
                for category in self._output_categories:
                    for t in category_tensors[category]:
                        if type(t) is TensorGPU:
                            category_device[category].append(mx_gpu_device)
                        else:
                            category_device[category].append(mx_cpu_device)
                d = []
                l = []
                for j, (shape, dtype) in enumerate(category_info[DALIGenericIterator.DATA_TAG]):
                    d.append(get_mx_array(shape, category_device[DALIGenericIterator.DATA_TAG][j], dtype = dtype))
                for j, (shape, dtype) in enumerate(category_info[DALIGenericIterator.LABEL_TAG]):
                    l.append(get_mx_array(shape, category_device[DALIGenericIterator.LABEL_TAG][j], dtype = dtype))

                self._data_batches[i][self._current_data_batch] = mx.io.DataBatch(data=d, label=l)

            d = self._data_batches[i][self._current_data_batch].data
            l = self._data_batches[i][self._current_data_batch].label
            # Copy data from DALI Tensors to MXNet NDArrays
            if self._dynamic_shape:
                for j, (shape, dtype) in enumerate(category_info[DALIGenericIterator.DATA_TAG]):
                    if list(d[j].shape) != shape:
                        d[j] = get_mx_array(shape, d[j].context, dtype = dtype)
                for j, (shape, dtype) in enumerate(category_info[DALIGenericIterator.LABEL_TAG]):
                    if list(l[j].shape) != shape:
                        l[j] = get_mx_array(shape, l[j].context, dtype = dtype)

            for j, d_arr in enumerate(d):
                feed_ndarray(category_tensors[DALIGenericIterator.DATA_TAG][j], d_arr)
            for j, l_arr in enumerate(l):
                feed_ndarray(category_tensors[DALIGenericIterator.LABEL_TAG][j], l_arr)

        self._schedule_runs()

        self._advance_and_check_drop_last()

        copy_db_index = self._current_data_batch
        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                left = [self.batch_size - l for l in left]
                for i, to_pad in zip(range(self._num_gpus), left):
                    self._data_batches[i][copy_db_index].pad = to_pad
            else:
                for batch in self._data_batches:
                    batch[copy_db_index].pad = 0

        else:
            # padding the last batch
            if self._last_batch_policy == LastBatchPolicy.PARTIAL and (self._counter > self._size) and self._size > 0:
                # this is the last batch and we need to pad
                overflow = self._counter - self._size
                overflow_per_device = overflow // self._num_gpus
                difference = self._num_gpus - (overflow % self._num_gpus)
                for i in range(self._num_gpus):
                    if i < difference:
                        self._data_batches[i][copy_db_index].pad = overflow_per_device
                    else:
                        self._data_batches[i][copy_db_index].pad = overflow_per_device + 1
            else:
                for db in self._data_batches:
                    db[copy_db_index].pad = 0

        return [db[copy_db_index] for db in self._data_batches]

    DATA_TAG = "data"
    LABEL_TAG = "label"

class DALIClassificationIterator(DALIGenericIterator):
    """
    DALI iterator for classification tasks for MXNet. It returns 2 outputs
    (data and label) in the form of MXNet's DataBatch of NDArrays.

    Calling

    .. code-block:: python

       DALIClassificationIterator(pipelines, reader_name, data_name, label_name, data_layout)

    is equivalent to calling

    .. code-block:: python

       DALIGenericIterator(pipelines,
                           [(data_name, DALIClassificationIterator.DATA_TAG),
                            (label_name, DALIClassificationIterator.LABEL_TAG)],
                           reader_name,
                           data_layout)

    Please keep in mind that NDArrays returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another NDArray.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    size : int, default = -1
           Number of samples in the shard for the wrapped pipeline (if there is more than one it is a sum)
           Providing -1 means that the iterator will work until StopIteration is raised
           from the inside of iter_setup(). The options `last_batch_policy`, `last_batch_padded` and
           `auto_reset` don't work in such case. It works with only one pipeline inside
           the iterator.
           Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy` to
                PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    data_name : str, optional, default = 'data'
                Data name for provided symbols.
    label_name : str, optional, default = 'softmax_label'
                 Label name for provided symbols.
    data_layout : str, optional, default = 'NCHW'
                  Either 'NHWC' or 'NCHW' - layout of the pipeline outputs.
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    squeeze_labels: (DEPRECATED) bool, optional, default = True
                 Whether the iterator should squeeze the labels before
                 copying them to the ndarray.
                 This argument is deprecated and will be removed from future releases.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the mxnet.ndarray will be resized accordingly
                 if the shape of DALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy : default = FILL
                What to do with the last batch when there is no enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch (it doesn't literally drop but sets ``pad`` field of ndarray
                so the following code could use it to drop the data). If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = PARTIAL, last_batch_padded = True  -> last batch = ``[7]``, next iteration will return ``[1, 2]``

    last_batch_policy = PARTIAL, last_batch_padded = False -> last batch = ``[7]``, next iteration will return ``[2, 3]``

    last_batch_policy = FILL, last_batch_padded = True   -> last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    last_batch_policy = FILL, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``

    last_batch_policy = DROP, last_batch_padded = True   -> last batch = ``[5, 6]``, next iteration will return ``[1, 2]``

    last_batch_policy = DROP, last_batch_padded = False  -> last batch = ``[5, 6]``, next iteration will return ``[2, 3]``
    """
    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 data_name='data',
                 label_name='softmax_label',
                 data_layout='NCHW',
                 fill_last_batch=None,
                 auto_reset=False,
                 squeeze_labels=True,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL):
        super(DALIClassificationIterator, self).__init__(pipelines,
                                                         [(data_name, DALIClassificationIterator.DATA_TAG),
                                                          (label_name, DALIClassificationIterator.LABEL_TAG)],
                                                         size,
                                                         reader_name=reader_name,
                                                         data_layout=data_layout,
                                                         fill_last_batch=fill_last_batch,
                                                         auto_reset = auto_reset,
                                                         squeeze_labels=squeeze_labels,
                                                         dynamic_shape=dynamic_shape,
                                                         last_batch_padded = last_batch_padded,
                                                         last_batch_policy = last_batch_policy)

###############################################
###############################################
################## Gluon API ##################
###############################################
###############################################


class SmartArray(object):
    def __init__(self, array):
        self._data = array.reshape(-1)
        self._view = array

    def resize(self, shape):
        new_size = np.prod(shape)

        if new_size > self._data.size:
            self._data = get_mx_array(new_size, self._data.context, dtype=self._data.dtype)
            self._view = self._data
        elif new_size < self._data.size:
            self._view = self._data[:new_size]
        else:
            self._view = self._data
        self._view = self._view.reshape(shape)
        return self._view

    @property
    def view(self):
        return self._view


class DALIGluonIterator(_DALIMXNetIteratorBase):
    """
    General DALI iterator for MXNet with Gluon API. It can return any number of
    outputs from the DALI pipeline in the form of per GPU tuples. These tuples consisting of
    NDArrays (for outputs marked as DALIGluonIterator.DENSE_TAG) and list of NDArrays (for
    output marked as DALIGluonIterator.SPARSE_TAG).

    Please keep in mind that NDArrays returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another NDArray.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
            List of pipelines to use
    size : int, default = -1
            Number of samples in the shard for the wrapped pipeline (if there is more than one it is a sum)
            Providing -1 means that the iterator will work until StopIteration is raised
            from the inside of iter_setup(). The options `last_batch_policy`, `last_batch_padded` and
            `auto_reset` don't work in such case. It works with only one pipeline inside
            the iterator.
            Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy` to
                PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    output_types : list of str, optional, default = None
            List of tags indicating whether the pipeline(s) output batch is
            uniform (all the samples have the same size) or not. Batch output marked
            as the former will be returned as a single NDArray, the latter
            will be returned as a list of NDArray.
            Must be either DALIGluonIterator.DENSE_TAG
            or DALIGluonIterator.SPARSE_TAG.
            Length of output_types must match the number of output of the pipeline(s).
            If not set, all outputs are considered to be marked with DALIGluonIterator.DENSE_TAG.
    auto_reset : bool, optional, default = False
            Whether the iterator resets itself for the next epoch
            or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy : default = FILL
                What to do with the last batch when there is no enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch (it doesn't literally drop but sets ``pad`` field of ndarray
                so the following code could use it to drop the data). If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = PARTIAL, last_batch_padded = True  -> last batch = ``[7]``, next iteration will return ``[1, 2]``

    last_batch_policy = PARTIAL, last_batch_padded = False -> last batch = ``[7]``, next iteration will return ``[2, 3]``

    last_batch_policy = FILL, last_batch_padded = True   -> last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    last_batch_policy = FILL, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``

    last_batch_policy = DROP, last_batch_padded = True   -> last batch = ``[5, 6]``, next iteration will return ``[1, 2]``

    last_batch_policy = DROP, last_batch_padded = False  -> last batch = ``[5, 6]``, next iteration will return ``[2, 3]``
    """
    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 output_types=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL):

        # check the assert first as _DaliBaseIterator would run the prefetch
        self._output_tags = {DALIGluonIterator.DENSE_TAG, DALIGluonIterator.SPARSE_TAG}
        assert output_types is None or set(output_types) <= self._output_tags, \
            "Only DENSE_TAG and SPARSE_TAG are allowed"

        self._outputs_types = output_types

        super(DALIGluonIterator, self).__init__(
            pipelines,
            size,
            reader_name,
            fill_last_batch,
            last_batch_padded,
            auto_reset,
            last_batch_policy)

        self._data_batches = [None for i in range(self._num_gpus)]

        self._first_batch = None
        try:
            self._first_batch = self._first_batch = DALIGluonIterator.__next__(self)
        except StopIteration:
            assert False, "It seems that there is no data in the pipeline. This may happen if `last_batch_policy` is set to PARTIAL and the requested batch size is greater than the shard size."

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        dali_outputs = self._get_outputs()

        for i in range(self._num_gpus):
            output_elements = []
            shapes = []
            for j, out in enumerate(dali_outputs[i]):
                if self._outputs_types is None or self._outputs_types[j] == DALIGluonIterator.DENSE_TAG:
                    output_elements.append(out.as_tensor())
                    shapes.append(output_elements[-1].shape())
                else:
                    output_elements.append([out[sample_idx] for sample_idx in range(self.batch_size)])
                    s = [t.shape() for t in output_elements[-1]]
                    shapes.append(s)

            if self._data_batches[i] is None:
                self._data_batches[i] = self._create_data_batch(output_elements, shapes, self._pipes[i].device_id)

            batch = self._data_batches[i]
            # Copy data from DALI Tensors to MXNet NDArrays
            for j, output_el in enumerate(output_elements):
                if self._outputs_types is None or self._outputs_types[j] == DALIGluonIterator.DENSE_TAG:
                    ndarray = batch[j].resize(shapes[j])
                    feed_ndarray(output_el, ndarray)
                else:
                    for sample_idx in range(self.batch_size):
                        ndarray = batch[j][sample_idx].resize(shapes[j][sample_idx])
                        feed_ndarray(output_el[sample_idx], ndarray)

        batches = [[([sample.view for sample in output_el] if isinstance(output_el,list) else output_el.view)
                    for output_el in batch]
                   for batch in self._data_batches]

        self._schedule_runs()

        self._advance_and_check_drop_last()

        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                output = []
                for batch, to_copy in zip(batches, left):
                    batch = batch.copy()
                    for element_idx in range(len(batch)):
                        batch[element_idx] = batch[element_idx][0:to_copy]
                    output.append(batch)
                return output

        else:
            if self._last_batch_policy == LastBatchPolicy.PARTIAL and (self._counter > self._size) and self._size > 0:
                # First calculate how much data is required to return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
                # Figure out how many GPUs to grab from.
                numGPUs_tograb = int(np.ceil(diff/self.batch_size))
                # Figure out how many results to grab from the last GPU (as a fractional GPU batch may be required to
                # bring us right up to self._size).
                mod_diff = diff % self.batch_size
                data_fromlastGPU = mod_diff if mod_diff else self.batch_size

                # Grab the relevant data.
                # 1) Grab everything from the relevant GPUs.
                # 2) Grab the right data from the last GPU.
                # 3) Append data together correctly and return.
                output = batches[0:numGPUs_tograb]
                output[-1] = output[-1].copy()
                for element_idx in range(len(output[-1])):
                    output[-1][element_idx] = output[-1][element_idx][0:data_fromlastGPU]
                return output

        return batches

    def _create_data_batch(self, output_elements, shapes, device_id):
        mx_gpu_device = mx.gpu(device_id)
        mx_cpu_device = mx.cpu(0)
        new_batch = []
        for j, output_el in enumerate(output_elements):
            first_t = output_el if self._outputs_types is None or self._outputs_types[j] == DALIGluonIterator.DENSE_TAG else output_el[0]
            dtype = np.dtype(first_t.dtype())
            device = mx_gpu_device if type(first_t) is TensorGPU else mx_cpu_device
            if self._outputs_types is None or self._outputs_types[j] == DALIGluonIterator.DENSE_TAG:
                new_batch.append(SmartArray(get_mx_array(shapes[j], device, dtype=dtype)))
            else:
                l = []
                for sample_idx in range(self.batch_size):
                    l.append(SmartArray(get_mx_array(shapes[j][sample_idx], device, dtype=dtype)))
                new_batch.append(l)
        return new_batch

    DENSE_TAG = "dense"
    SPARSE_TAG = "sparse"
