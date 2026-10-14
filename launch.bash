#!/bin/bash
# 使い方: ./launch.bash [対象ディレクトリ]
#   引数を省略すると TARGET_DIR 以下の rosbag を処理する。どこから実行してもよい。

TARGET_DIR="$HOME/raspicat2/rosbags/chain/tsudanuma/tooi/dataset_circuit_v2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$TARGET_DIR}"

exec python3 "$SCRIPT_DIR/bag_dataset_maker.py" "$TARGET"

