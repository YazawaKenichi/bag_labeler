"""rosbag2 (.db3) からデータセットを作るためのコア処理 (GUI 非依存)."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

IMAGE_TYPES = ('sensor_msgs/msg/CompressedImage', 'sensor_msgs/msg/Image')
# 数値データとして扱わないトピック型
SKIP_TYPES = (
    'rcl_interfaces/msg/Log',
    'rcl_interfaces/msg/ParameterEvent',
    'diagnostic_msgs/msg/DiagnosticArray',
    'tf2_msgs/msg/TFMessage',
    'lifecycle_msgs/msg/TransitionEvent',
    'bond/msg/Status',
    'rosbag2_interfaces/msg/WriteSplitEvent',
)
# これより長い数値配列は展開しない (点群・LaserScan 等で列が爆発するのを防ぐ)
MAX_ARRAY_LEN = 256

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def find_bags(root: Path) -> list[Path]:
    """root 以下で metadata.yaml と .db3 を含むディレクトリを列挙する."""
    bags = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if 'metadata.yaml' in filenames and any(f.endswith('.db3') for f in filenames):
            bags.append(Path(dirpath))
    return sorted(bags)


def flatten(value, prefix: str, out: dict[str, float]) -> None:
    """メッセージを再帰的にたどり、数値フィールドを {名前: 値} に平坦化する."""
    msgtype = getattr(value, '__msgtype__', None)
    if msgtype is not None:
        if msgtype == 'builtin_interfaces/msg/Time':
            out[prefix] = value.sec + value.nanosec * 1e-9
            return
        for name in value.__dataclass_fields__:
            flatten(getattr(value, name), f'{prefix}.{name}' if prefix else name, out)
    elif isinstance(value, (bool, int, float, np.number, np.bool_)):
        out[prefix] = float(value)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in 'biuf' and value.size <= MAX_ARRAY_LEN:
            for i, x in enumerate(value.ravel()):
                out[f'{prefix}[{i}]'] = float(x)
    elif isinstance(value, (list, tuple)) and len(value) <= MAX_ARRAY_LEN:
        if all(isinstance(x, (bool, int, float, np.number)) for x in value):
            for i, x in enumerate(value):
                out[f'{prefix}[{i}]'] = float(x)


@dataclass
class TopicSeries:
    topic: str
    msgtype: str
    times: np.ndarray  # int64 ns (bag 記録時刻)
    fields: dict[str, np.ndarray]  # フィールド名 -> float64 配列 (times と同じ長さ)


@dataclass
class ImageSeries:
    topic: str
    msgtype: str
    times: np.ndarray  # int64 ns
    raw: list[bytes]  # 未デシリアライズの CDR


@dataclass
class BagData:
    path: Path
    start_ns: int
    end_ns: int
    series: dict[str, TopicSeries] = field(default_factory=dict)
    images: dict[str, ImageSeries] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name


def load_bag(path: Path, progress=None) -> BagData:
    """bag を読み込み、数値トピックは平坦化、画像トピックは生データのまま保持する."""
    with AnyReader([path], default_typestore=TYPESTORE) as reader:
        bag = BagData(path, reader.start_time, reader.end_time)
        conns = [c for c in reader.connections
                 if c.msgcount > 0 and c.msgtype not in SKIP_TYPES]
        total = sum(c.msgcount for c in conns) or 1
        times: dict[str, list[int]] = {}
        rows: dict[str, list[dict[str, float]]] = {}
        types: dict[str, str] = {}
        for n, (conn, t, raw) in enumerate(reader.messages(connections=conns)):
            if progress and n % 200 == 0:
                progress(n / total)
            topic = conn.topic
            types[topic] = conn.msgtype
            if conn.msgtype in IMAGE_TYPES:
                img = bag.images.setdefault(
                    topic, ImageSeries(topic, conn.msgtype, [], []))
                img.times.append(t)
                img.raw.append(bytes(raw))
                continue
            out: dict[str, float] = {}
            flatten(reader.deserialize(raw, conn.msgtype), '', out)
            if out:
                times.setdefault(topic, []).append(t)
                rows.setdefault(topic, []).append(out)
        for img in bag.images.values():
            order = np.argsort(img.times, kind='stable')
            img.times = np.asarray(img.times, dtype=np.int64)[order]
            img.raw = [img.raw[i] for i in order]
        for topic, ts in times.items():
            names: list[str] = []
            seen = set()
            for r in rows[topic]:
                for k in r:
                    if k not in seen:
                        seen.add(k)
                        names.append(k)
            order = np.argsort(ts, kind='stable')
            fields_ = {
                k: np.array([rows[topic][i].get(k, math.nan) for i in order])
                for k in names
            }
            bag.series[topic] = TopicSeries(
                topic, types[topic], np.asarray(ts, dtype=np.int64)[order], fields_)
    return bag


def decode_image(img: ImageSeries, index: int) -> np.ndarray | None:
    """BGR (H, W, 3) の uint8 画像を返す."""
    import cv2
    msg = TYPESTORE.deserialize_cdr(img.raw[index], img.msgtype)
    if img.msgtype == 'sensor_msgs/msg/CompressedImage':
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, np.uint8)
    if enc in ('bgr8', 'rgb8'):
        arr = data.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        return arr if enc == 'bgr8' else arr[:, :, ::-1].copy()
    if enc in ('bgra8', 'rgba8'):
        arr = data.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR if enc == 'bgra8' else cv2.COLOR_RGBA2BGR)
    if enc in ('mono8', '8uc1'):
        arr = data.reshape(h, msg.step)[:, :w]
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if enc in ('mono16', '16uc1'):
        arr = data.view(np.uint16).reshape(h, msg.step // 2)[:, :w]
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return None


# ---------------------------------------------------------------- ラベル区間

@dataclass
class Interval:
    start_ns: int
    stop_ns: int
    label: int = 1
    stop_source: str = 'manual'  # 'manual' | 'joy' | 'odom' | 'bag_end' (未設定)
    start_source: str = 'manual'  # 'manual' | 'after_prev' (一覧で直前の区間の終了に連動)


def rising_edges(series: TopicSeries, key: str, threshold: float) -> np.ndarray:
    """key の値が threshold 未満から以上に変わった時刻 (ns) の配列."""
    v = series.fields.get(key)
    if v is None:
        return np.array([], dtype=np.int64)
    pressed = np.nan_to_num(v, nan=-math.inf) >= threshold
    prev = np.concatenate([[True], pressed[:-1]])  # 先頭で押されていてもエッジとしない
    return series.times[pressed & ~prev]


def find_stop(series: TopicSeries | None, key: str, threshold: float,
              start_ns: int) -> int | None:
    """start_ns より後で最初に停止ボタンが押された (立ち上がった) 時刻.

    開始時点で既に押されている場合は、一度離されて再度押されるまで待つ。
    """
    if series is None:
        return None
    edges = rising_edges(series, key, threshold)
    after = edges[edges > start_ns]
    return int(after[0]) if after.size else None


def labels_path(bag_dir: Path) -> Path:
    return bag_dir / f'{bag_dir.name}_labels.json'


def csv_path(bag_dir: Path) -> Path:
    return bag_dir / f'{bag_dir.name}_dataset.csv'


def load_labels(bag_dir: Path) -> tuple[list[Interval], float | None]:
    """ラベル区間と映像オフセット [s] (未保存なら None) を読む."""
    p = labels_path(bag_dir)
    if not p.exists():
        return [], None
    data = json.loads(p.read_text())
    return [Interval(**d) for d in data.get('intervals', [])], data.get('video_offset')


def save_labels(bag_dir: Path, intervals: list[Interval], video_offset: float) -> None:
    data = {'video_offset': video_offset,
            'intervals': [dataclasses.asdict(iv) for iv in intervals]}
    labels_path(bag_dir).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def stop_events(series: TopicSeries | None, keys: list[str],
                still: float = 0.005, moving: float = 0.05) -> np.ndarray:
    """速度 (keys の絶対値の最大) が moving 以上の走行状態から still 未満になった時刻 (ns)."""
    if series is None:
        return np.array([], dtype=np.int64)
    cols = [np.abs(series.fields[k]) for k in keys if k in series.fields]
    if not cols:
        return np.array([], dtype=np.int64)
    speed = np.nan_to_num(np.max(cols, axis=0))
    events = []
    running = False
    for t, v in zip(series.times, speed):
        if v >= moving:
            running = True
        elif running and v < still:
            events.append(t)
            running = False
    return np.asarray(events, dtype=np.int64)


# ---------------------------------------------------------------- データセット生成

def column_name(topic: str, fld: str) -> str:
    return f'{topic}/{fld}'


def build_dataset(bag: BagData, ref_topic: str, columns: list[tuple[str, str]],
                  intervals: list[Interval]):
    """基準トピックの受信時刻を行とし、他トピックは直前値保持で埋める.

    戻り値: (header, times_ns, values (N x len(columns)), labels (N,))
    """
    ref = bag.series[ref_topic]
    t = ref.times
    values = np.full((len(t), len(columns)), math.nan)
    for j, (topic, fld) in enumerate(columns):
        s = bag.series.get(topic)
        if s is None or fld not in s.fields:
            continue
        idx = np.searchsorted(s.times, t, side='right') - 1
        ok = idx >= 0
        values[ok, j] = s.fields[fld][idx[ok]]
    labels = np.zeros(len(t), dtype=np.int64)
    for iv in intervals:  # 後から追加した区間が優先
        labels[(t >= iv.start_ns) & (t < iv.stop_ns)] = iv.label
    header = ['elapsed'] +[column_name(*c) for c in columns] + ['label']
    return header, t, values, labels


def write_csv(path: Path, bag: BagData, header, times, values, labels) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(times)):
            row = [f'{(times[i] - bag.start_ns) * 1e-9:.6f}']  # bag 先頭からの秒
            row += ['' if math.isnan(v) else f'{v:.10g}' for v in values[i]]
            row.append(int(labels[i]))
            w.writerow(row)
