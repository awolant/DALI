# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest


class TestDaliVideoPluginLoadOk(unittest.TestCase):
    def test_import_dali_video_ok(self):
        import nvidia.dali.plugin.video as dali_video  # noqa: F401
        assert True


class TestDaliVideoPluginLoadFail(unittest.TestCase):
    def test_import_dali_video_load_fail(self):
        with self.assertRaises(Exception):
            import nvidia.dali.plugin.video as dali_video  # noqa: F401


if __name__ == "__main__":
    unittest.main()