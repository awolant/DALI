#!/bin/bash -e

test_nose() {
    # readers and decoders are tested in the TL0_python-self-test-core
    FILTER_PATTERN="test_operator_readers_.*\.py\|test_operator_decoders_.*\.py"

    for test_script in $(ls test_operator_*.py | sed "/$FILTER_PATTERN/d"); do
        ${python_test_runner} ${python_test_args} --attr '!slow,!pytorch,!mxnet,!cupy,!numba' ${test_script}
    done
}

test_no_fw() {
    test_nose
}

run_all() {
  test_no_fw
}
