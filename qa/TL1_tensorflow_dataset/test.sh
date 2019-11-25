#!/bin/bash -e
# used pip packages
pip_packages="nose tensorflow-gpu"
target_dir=./dali/test/python

# populate epilog and prolog with variants to enable/disable conda and virtual env
# every test will be executed for bellow configs
prolog=(: enable_conda enable_virtualenv)
epilog=(: disable_conda disable_virtualenv)

test_body() {
    # Manually removing the supported plugin so that it fails
    lib_dir=$(python -c 'import nvidia.dali.sysconfig as sc; print(sc.get_lib_dir())')
    rm -rf $lib_dir/plugin/*.so

    # Installing "current" dali tf (built against installed TF)
    pip install ../../../nvidia-dali-tf-plugin*.tar.gz

    is_compatible=$(python -c 'import nvidia.dali.plugin.tf as dali_tf; print(dali_tf.dataset_compatible_tensorflow())')
    if [ $is_compatible = 'True' ]; then
        # DALI TF DATASET run
        nosetests --verbose -s test_dali_tf_dataset.py:_test_tf_dataset_other_gpu
        nosetests --verbose -s test_dali_tf_dataset.py:_test_tf_dataset_multigpu
        nosetests --verbose -s test_dali_tf_dataset_mnist.py

        # DALI TF Notebooks run
        find ../../../docs/examples/tensorflow/* -name "tensorflow-dataset*.ipynb" | xargs -i jupyter nbconvert \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600 {}
    fi
}

pushd ../..
source ./qa/test_template.sh
popd
