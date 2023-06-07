# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from pathlib import Path

from .local import Controller

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: _local.py <submitit_folder>"
    # most arguments are read from environment variables.
    controller = Controller(Path(sys.argv[1]))
    controller.run()
