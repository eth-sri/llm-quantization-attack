# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum


class Analyzer(str, enum.Enum):
    REGEX = "regex"
    WEGGLI = "weggli"
    SEMGREP = "semgrep"

    def __str__(self) -> str:
        return self.name.lower()
