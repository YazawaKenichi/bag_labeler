#!/usr/bin/env python3
"""rosbag2 (.db3) を映像で確認しながらラベル付けし、CSV データセットを作る GUI.

使い方:
    python3 bag_labeler.py [探索ルート (既定: カレントディレクトリ)]

キー操作:
    ← / →            1 フレーム移動 (Shift で 10 フレーム)
    Home / End       先頭 / 末尾
    Space            再生 / 一時停止
    Enter            現在フレームからラベル区間を開始. 終了は「区間の終了」設定による.
                     既定では odom 停止までを label、その後 bag 末尾までを label 2 にする
    E                選択中 (未選択なら最後) の区間の終了を現在フレームにする
    Delete           選択中の区間を削除
    Ctrl+S           CSV 保存
"""

from __future__ import annotations

import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QEvent, QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressDialog, QPushButton, QShortcut, QSizePolicy, QSlider, QSpinBox, QSplitter,
    QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

import bag_core as bc

CONFIG_NAME = '.bag_labeler.json'
# 映像同期 (「odom 停止に合わせる」ボタン) で停止時刻を求める速度トピック
SPEED_TOPIC = '/odom'
SPEED_FIELDS = ['twist.twist.linear.x', 'twist.twist.angular.z']
DEFAULT_PLOTS = ['/hx711/raw|data', '/cmd_vel|linear.x', '/cmd_vel|angular.z', '/joy|buttons[5]']
PLOT_WINDOWS = [('全体', 0.0), ('2 s', 2.0), ('5 s', 5.0), ('10 s', 10.0), ('20 s', 20.0)]
SIGNAL_COLORS = [QColor(129, 212, 250), QColor(255, 183, 77), QColor(174, 213, 129),
                 QColor(240, 98, 146), QColor(179, 157, 219), QColor(77, 208, 225)]
# odom 停止後 (接触したまま停止している間) に付けるラベル
AFTER_STOP_LABEL = 2
# 区間の終了の決め方 (表示名, モード)
END_MODES = [(f'odom 停止まで → 以降 label {AFTER_STOP_LABEL}', 'odom_then_after'),
             ('手動 (E キー)', 'manual'), ('停止ボタン', 'joy'), ('odom 停止', 'odom')]
STOP_SOURCE_NAMES = {'joy': '停止ボタン', 'odom': 'odom 停止', 'manual': '手動',
                     'bag_end': '未設定 (bag 末尾まで)'}
INTERVAL_COLORS = [QColor(66, 165, 245, 90), QColor(255, 167, 38, 90),
                   QColor(102, 187, 106, 90), QColor(171, 71, 188, 90)]
# 映像の表示調整 (設定キー, 表示名, 最小, 最大, 既定, 値の表示). プレビューにだけ反映する
VIEW_ADJUST = [('view_brightness', '明るさ', -100, 100, 0, lambda v: f'{v:+d}'),
               ('view_contrast', 'コントラスト', 0, 300, 100, lambda v: f'{v} %'),
               ('view_gamma', 'ガンマ', 10, 300, 100, lambda v: f'{v / 100:.2f}')]


def adjust_lut(brightness: int, contrast: int, gamma: int) -> np.ndarray | None:
    """明るさ・コントラスト・ガンマを 8bit の変換表にする. 無変換なら None.

    ガンマ (1 より大きいと中間調が明るくなる) → コントラスト (中央の灰色を中心に拡大)
    → 明るさ (全体に加算) の順に適用する。
    """
    if (brightness, contrast, gamma) == (0, 100, 100):
        return None
    x = np.arange(256) / 255.0
    y = x ** (100.0 / gamma)
    y = (y - 0.5) * contrast / 100.0 + 0.5 + brightness / 100.0
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


class Timeline(QWidget):
    """波形・ラベル区間・停止ボタン押下位置を表示し、クリック/ドラッグでシークする."""

    seek = pyqtSignal(int)  # ns

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.t0, self.t1 = 0, 1
        self.wave_t = np.array([])
        self.wave_v = np.array([])
        self.wave_name = ''
        self.edges = np.array([])
        self.stops = np.array([])
        self.intervals: list[bc.Interval] = []
        self.selected = -1
        self.cursor_ns = 0

    def x_of(self, t) -> float:
        return (t - self.t0) / max(self.t1 - self.t0, 1) * self.width()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(30, 30, 30))
        for i, iv in enumerate(self.intervals):
            c = QColor(INTERVAL_COLORS[(iv.label - 1) % len(INTERVAL_COLORS)])
            if i == self.selected:
                c.setAlpha(170)
            x0, x1 = self.x_of(iv.start_ns), self.x_of(iv.stop_ns)
            p.fillRect(QRectF(x0, 0, max(x1 - x0, 1), h), c)
            p.setPen(QColor(230, 230, 230))
            p.drawText(QRectF(x0 + 3, 2, 200, 14), Qt.AlignLeft, f'#{i + 1} L{iv.label}')
        v = self.wave_v
        ok = ~np.isnan(v) if v.size else v
        if v.size and ok.any():
            lo, hi = np.nanmin(v), np.nanmax(v)
            span = hi - lo or 1.0
            path = QPainterPath()
            first = True
            for t, y in zip(self.wave_t[ok], v[ok]):
                pt = (self.x_of(t), h - 8 - (y - lo) / span * (h - 26))
                if first:
                    path.moveTo(*pt)
                    first = False
                else:
                    path.lineTo(*pt)
            p.setPen(QPen(QColor(129, 212, 250), 1.2))
            p.drawPath(path)
            p.setPen(QColor(160, 160, 160))
            p.drawText(QRectF(4, h - 16, w - 8, 14), Qt.AlignRight,
                       f'{self.wave_name}  [{lo:.4g}, {hi:.4g}]')
        p.setPen(QPen(QColor(239, 83, 80), 2))
        for t in self.edges:
            x = self.x_of(t)
            p.drawLine(int(x), 0, int(x), 12)
        p.setPen(QPen(QColor(102, 187, 106), 2))
        for t in self.stops:
            x = self.x_of(t)
            p.drawLine(int(x), h - 12, int(x), h)
        p.setPen(QPen(QColor(255, 235, 59), 2))
        x = self.x_of(self.cursor_ns)
        p.drawLine(int(x), 0, int(x), h)

    def _emit(self, x):
        frac = min(max(x / max(self.width(), 1), 0.0), 1.0)
        self.seek.emit(int(self.t0 + frac * (self.t1 - self.t0)))

    def mousePressEvent(self, e):
        self._emit(e.x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._emit(e.x())


class PlotPanel(QWidget):
    """複数の時系列を縦に並べて表示する. 直前値保持 (CSV と同じ) の階段状に描く.

    左の欄に、カーソル (= 表示中の映像フレーム) 時点の値を表示する。
    """

    seek = pyqtSignal(int)  # ns
    AXIS_H = 22
    LABEL_W = 230
    MIN_STRIP_H = 38

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.signals: list[tuple[str, np.ndarray, np.ndarray]] = []  # (名前, times, values)
        self.t0, self.t1 = 0, 1
        self.window = 0.0  # 表示幅 [s]. 0 で bag 全体
        self.cursor_ns = 0
        self.intervals: list[bc.Interval] = []
        self.edges = np.array([])
        self.stops = np.array([])

    def set_signals(self, signals):
        self.signals = signals
        self.setMinimumHeight(max(120, len(signals) * self.MIN_STRIP_H + self.AXIS_H))
        self.update()

    def view_range(self) -> tuple[int, int]:
        if self.window <= 0 or self.window * 1e9 >= self.t1 - self.t0:
            return self.t0, self.t1
        half = int(self.window * 1e9 / 2)
        a = min(max(self.cursor_ns - half, self.t0), self.t1 - 2 * half)
        return a, a + 2 * half

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(24, 24, 24))
        if not self.signals:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignCenter,
                       '右のツリーの「プロット」列にチェックすると表示されます')
            return
        a, b = self.view_range()
        span = max(b - a, 1)
        L = self.LABEL_W
        pw = max(w - L, 1)

        def x_of(t):
            return L + (t - a) / span * pw

        plot_h = h - self.AXIS_H
        p.fillRect(QRectF(0, 0, L, h), QColor(36, 36, 36))
        p.setClipRect(QRectF(L, 0, pw, h))
        for iv in self.intervals:
            c = QColor(INTERVAL_COLORS[(iv.label - 1) % len(INTERVAL_COLORS)])
            c.setAlpha(55)
            x0, x1 = x_of(iv.start_ns), x_of(iv.stop_ns)
            p.fillRect(QRectF(x0, 0, max(x1 - x0, 1), plot_h), c)
        for ts, color in ((self.edges, QColor(239, 83, 80)), (self.stops, QColor(102, 187, 106))):
            p.setPen(QPen(color, 1, Qt.DashLine))
            for t in ts:
                if a <= t <= b:
                    p.drawLine(int(x_of(t)), 0, int(x_of(t)), plot_h)
        p.setClipping(False)

        n = len(self.signals)
        sh = plot_h / n
        name_font = p.font()
        value_font = p.font()
        value_font.setBold(True)
        value_font.setPointSizeF(name_font.pointSizeF() * 1.25)
        p.setRenderHint(QPainter.Antialiasing)
        for k, (name, times, vals) in enumerate(self.signals):
            top = k * sh
            color = SIGNAL_COLORS[k % len(SIGNAL_COLORS)]
            p.setPen(QColor(60, 60, 60))
            p.drawLine(0, int(top + sh), w, int(top + sh))
            j = int(np.searchsorted(times, self.cursor_ns, side='right')) - 1
            cur = vals[j] if j >= 0 else math.nan
            if (~np.isnan(vals)).any():
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
                vmin, vmax = lo, hi
                if hi == lo:
                    lo, hi = lo - 1, hi + 1
                pad = (hi - lo) * 0.1
                lo, hi = lo - pad, hi + pad

                def y_of(v):
                    return top + 3 + (hi - v) / (hi - lo) * (sh - 6)
                i0 = max(int(np.searchsorted(times, a, side='right')) - 1, 0)
                i1 = min(int(np.searchsorted(times, b, side='right')) + 1, len(times))
                path = QPainterPath()
                started = False
                prev_y = 0.0
                for t, v in zip(times[i0:i1], vals[i0:i1]):
                    if np.isnan(v):
                        started = False
                        continue
                    x, y = max(x_of(t), L), y_of(v)
                    if not started:
                        path.moveTo(x, y)
                        started = True
                    else:
                        path.lineTo(x, prev_y)
                        path.lineTo(x, y)
                    prev_y = y
                if started:
                    path.lineTo(x_of(b), prev_y)
                p.setPen(QPen(color, 1.4))
                p.drawPath(path)
                if not np.isnan(cur):
                    p.setBrush(color)
                    p.drawEllipse(QRectF(x_of(self.cursor_ns) - 3, y_of(cur) - 3, 6, 6))
                    p.setBrush(Qt.NoBrush)
                p.setPen(QColor(110, 110, 110))
                p.drawText(QRectF(w - 90, top + 1, 86, 14), Qt.AlignRight, f'{vmax:.4g}')
                p.drawText(QRectF(w - 90, top + sh - 15, 86, 14), Qt.AlignRight, f'{vmin:.4g}')
            # 左欄: 名前と現在値
            p.setFont(name_font)
            p.setPen(QColor(170, 170, 170))
            p.drawText(QRectF(6, top + 2, L - 10, sh / 2), Qt.AlignLeft | Qt.AlignTop, name)
            p.setFont(value_font)
            p.setPen(color)
            p.drawText(QRectF(6, top, L - 10, sh - 2), Qt.AlignRight | Qt.AlignBottom,
                       '-' if np.isnan(cur) else f'{cur:.6g}')
            p.setFont(name_font)

        # 時間軸 (bag 先頭からの秒)
        p.setPen(QColor(150, 150, 150))
        sec = span * 1e-9
        step = next((s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600)
                     if sec / s <= pw / 70), 1200)
        t_s = math.ceil((a - self.t0) * 1e-9 / step) * step
        while t_s <= (b - self.t0) * 1e-9 + 1e-9:
            x = x_of(self.t0 + t_s * 1e9)
            p.drawLine(int(x), plot_h, int(x), plot_h + 4)
            p.drawText(QRectF(x - 30, plot_h + 3, 60, 14), Qt.AlignCenter, f'{t_s:.10g}')
            t_s += step
        p.drawText(QRectF(6, plot_h + 3, L - 10, 14), Qt.AlignLeft,
                   f'現在 {(self.cursor_ns - self.t0) * 1e-9:.3f} s')
        p.setPen(QPen(QColor(255, 235, 59), 2))
        x = x_of(self.cursor_ns)
        p.drawLine(int(x), 0, int(x), plot_h)

    def _emit(self, x):
        a, b = self.view_range()
        frac = min(max((x - self.LABEL_W) / max(self.width() - self.LABEL_W, 1), 0.0), 1.0)
        self.seek.emit(int(a + frac * (b - a)))

    def mousePressEvent(self, e):
        self._emit(e.x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._emit(e.x())


class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.config_path = root / CONFIG_NAME
        self.config = self._load_config()
        self.bag: bc.BagData | None = None
        self.bag_dir: Path | None = None
        self.intervals: list[bc.Interval] = []
        self.video_offset = 0.0  # [s] 映像の記録時刻 - 実際の時刻 (カメラ遅延)
        self.frame = 0
        self.img_cache: dict[int, np.ndarray] = {}  # デコード済み RGB (表示調整前)
        self.view_lut: np.ndarray | None = None
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)
        self.play_origin = (0.0, 0)
        self._building_tree = False

        self.setWindowTitle(f'Bag Labeler - {root}')
        self.resize(1500, 900)
        self._build_ui()
        self._scan_bags()
        QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------ 設定

    def _load_config(self) -> dict:
        defaults = {'selected': [], 'ref_topic': '/hx711/raw', 'image_topic': '',
                    'joy_topic': '/joy', 'joy_kind': 'buttons', 'joy_index': 5,
                    'joy_threshold': 0.5, 'label': 1, 'plot_field': '', 'video_offset': 0.0, 'end_mode': 'odom_then_after',
                    'plots': list(DEFAULT_PLOTS), 'plot_window': 0.0, 'plot_open': True}
        defaults.update({key: default for key, _, _, _, default, _ in VIEW_ADJUST})
        try:
            defaults.update(json.loads(self.config_path.read_text()))
        except (OSError, ValueError):
            pass
        return defaults

    def _save_config(self):
        try:
            self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False))
        except OSError as e:
            self.statusBar().showMessage(f'設定を保存できません: {e}')

    # ------------------------------------------------------------ UI 構築

    def _build_ui(self):
        # 左: bag 一覧
        self.bag_list = QListWidget()
        self.bag_list.currentRowChanged.connect(self._on_bag_selected)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel('rosbag 一覧 (✓ = CSV 保存済み)'))
        lv.addWidget(self.bag_list)
        rescan = QPushButton('再スキャン')
        rescan.clicked.connect(self._scan_bags)
        lv.addWidget(rescan)

        # 中央: 映像 + シーク
        self.video = QLabel('bag を選択してください')
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(480, 270)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet('background:#111; color:#888;')
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged.connect(self._set_frame)
        self.timeline = Timeline()
        self.timeline.seek.connect(self._seek_ns)
        self.info = QLabel()
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_play = QPushButton('▶ 再生 (Space)')
        self.btn_play.clicked.connect(self._toggle_play)
        btn_start = QPushButton('ここから開始 (Enter)')
        btn_start.clicked.connect(self._add_interval)
        btn_end = QPushButton('ここで終了 (E)')
        btn_end.clicked.connect(self._set_end_here)
        btn_del = QPushButton('区間削除 (Del)')
        btn_del.clicked.connect(self._delete_interval)
        btn_save = QPushButton('CSV 保存 (Ctrl+S)')
        btn_save.clicked.connect(self._save_csv)
        btn_save.setStyleSheet('font-weight:bold;')
        self.label_spin = QSpinBox()
        self.label_spin.setRange(1, 99)
        self.label_spin.setValue(int(self.config['label']))
        self.label_spin.setPrefix('label=')
        self.label_spin.valueChanged.connect(lambda v: self._cfg('label', v))
        bh = QHBoxLayout()
        for wdg in (self.btn_play, btn_start, self.label_spin, btn_end, btn_del):
            bh.addWidget(wdg)
        bh.addStretch()
        bh.addWidget(btn_save)

        self.interval_list = QListWidget()
        self.interval_list.setMaximumHeight(120)
        self.interval_list.currentRowChanged.connect(self._on_interval_selected)
        self.interval_list.itemDoubleClicked.connect(
            lambda it: self._seek_ns(self.intervals[self.interval_list.row(it)].start_ns))

        # プロット (折り畳み可)
        self.plot_panel = PlotPanel()
        self.plot_panel.seek.connect(self._seek_ns)
        self.plot_toggle = QToolButton()
        self.plot_toggle.setCheckable(True)
        self.plot_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.plot_toggle.setText('プロット')
        self.plot_toggle.setStyleSheet('QToolButton { border: none; font-weight: bold; }')
        self.plot_toggle.toggled.connect(self._toggle_plot)
        self.plot_window = QComboBox()
        for name, sec in PLOT_WINDOWS:
            self.plot_window.addItem(name, sec)
        idx = self.plot_window.findData(float(self.config['plot_window']))
        self.plot_window.setCurrentIndex(max(idx, 0))
        self.plot_window.currentIndexChanged.connect(self._on_plot_window)
        self.plot_panel.window = self.plot_window.currentData()
        ph = QHBoxLayout()
        ph.setContentsMargins(0, 0, 0, 0)
        ph.addWidget(self.plot_toggle)
        ph.addStretch()
        ph.addWidget(QLabel('表示幅'))
        ph.addWidget(self.plot_window)
        plot_box = QWidget()
        pv = QVBoxLayout(plot_box)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addLayout(ph)
        pv.addWidget(self.plot_panel, 1)
        self.view_split = QSplitter(Qt.Vertical)
        self.view_split.addWidget(self.video)
        self.view_split.addWidget(plot_box)
        self.view_split.setStretchFactor(0, 1)
        self.view_split.setStretchFactor(1, 1)
        self.view_split.splitterMoved.connect(lambda *_: self._show_current_image())
        self._plot_sizes = None

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.addWidget(self.view_split, 1)
        cv.addWidget(self.slider)
        cv.addWidget(self.timeline)
        cv.addWidget(self.info)
        cv.addLayout(bh)
        cv.addWidget(QLabel('ラベル区間 (ダブルクリックで開始位置へ)'))
        cv.addWidget(self.interval_list)

        # 右: 設定とフィールド選択
        self.image_combo = QComboBox()
        self.image_combo.currentTextChanged.connect(self._on_image_topic)
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-10, 10)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setSuffix(' s')
        self.offset_spin.setToolTip('映像を何秒前にずらしてデータと合わせるか (カメラ遅延)')
        self.offset_spin.valueChanged.connect(self._set_offset)
        btn_sync = QPushButton('odom 停止に合わせる')
        btn_sync.setFocusPolicy(Qt.NoFocus)
        btn_sync.setToolTip('映像でロボットが止まったフレームで押すと、'
                            'odom の停止時刻に合うよう映像オフセットを設定する')
        btn_sync.clicked.connect(self._sync_offset)
        oh = QHBoxLayout()
        oh.addWidget(self.offset_spin, 1)
        oh.addWidget(btn_sync)
        self.ref_combo = QComboBox()
        self.ref_combo.currentTextChanged.connect(self._on_ref_topic)
        self.plot_combo = QComboBox()
        self.plot_combo.currentTextChanged.connect(self._on_plot_field)
        self.joy_topic = QComboBox()
        self.joy_topic.setEditable(True)
        self.joy_topic.currentTextChanged.connect(lambda v: self._joy_changed('joy_topic', v))
        self.joy_kind = QComboBox()
        self.joy_kind.addItems(['buttons', 'axes'])
        self.joy_kind.setCurrentText(self.config['joy_kind'])
        self.joy_kind.currentTextChanged.connect(lambda v: self._joy_changed('joy_kind', v))
        self.joy_index = QSpinBox()
        self.joy_index.setRange(0, 63)
        self.joy_index.setValue(int(self.config['joy_index']))
        self.joy_index.valueChanged.connect(lambda v: self._joy_changed('joy_index', v))
        self.joy_thr = QDoubleSpinBox()
        self.joy_thr.setRange(-10, 10)
        self.joy_thr.setSingleStep(0.1)
        self.joy_thr.setValue(float(self.config['joy_threshold']))
        self.joy_thr.valueChanged.connect(lambda v: self._joy_changed('joy_threshold', v))
        jh = QHBoxLayout()
        jh.addWidget(self.joy_kind)
        jh.addWidget(QLabel('['))
        jh.addWidget(self.joy_index)
        jh.addWidget(QLabel('] ≥'))
        jh.addWidget(self.joy_thr)

        settings = QGroupBox('設定')
        form = QFormLayout(settings)
        form.addRow('映像トピック', self.image_combo)
        form.addRow('映像オフセット', oh)
        form.addRow('基準トピック (行)', self.ref_combo)
        form.addRow('波形表示', self.plot_combo)
        form.addRow('停止ボタン トピック', self.joy_topic)
        form.addRow('停止ボタン 条件', jh)
        self.end_mode = QComboBox()
        for name, mode in END_MODES:
            self.end_mode.addItem(name, mode)
        self.end_mode.setCurrentIndex(max(self.end_mode.findData(self.config['end_mode']), 0))
        self.end_mode.currentIndexChanged.connect(
            lambda _: self._cfg('end_mode', self.end_mode.currentData()))
        self.end_mode.setToolTip('Enter で区間を開始したときの終了の決め方。'
                                 '既に作った区間には影響しない')
        form.addRow('区間の終了', self.end_mode)

        view_box = QGroupBox('映像の表示調整 (プレビューのみ)')
        vform = QFormLayout(view_box)
        self.view_sliders = {}
        for key, name, lo, hi, _, fmt in VIEW_ADJUST:
            sl = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setFocusPolicy(Qt.NoFocus)
            sl.setValue(int(self.config[key]))
            sl.valueChanged.connect(self._on_view_adjust)
            val = QLabel()
            val.setMinimumWidth(50)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row = QHBoxLayout()
            row.addWidget(sl, 1)
            row.addWidget(val)
            vform.addRow(name, row)
            self.view_sliders[key] = (sl, val, fmt)
        btn_reset = QPushButton('リセット')
        btn_reset.setFocusPolicy(Qt.NoFocus)
        btn_reset.clicked.connect(self._reset_view_adjust)
        vform.addRow(btn_reset)
        self._on_view_adjust()

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText('フィールド絞り込み')
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['CSV 列 / フィールド', 'プロット / 型'])
        self.tree.setColumnWidth(0, 250)
        self.tree.itemChanged.connect(self._on_tree_changed)
        self.sel_label = QLabel()

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(settings)
        rv.addWidget(view_box)
        rv.addWidget(QLabel('左の列: CSV に出力するデータ軸 / 右の列: プロット表示'))
        rv.addWidget(self.filter_edit)
        rv.addWidget(self.tree, 1)
        rv.addWidget(self.sel_label)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(center)
        split.addWidget(right)
        split.setSizes([330, 800, 400])
        self.setCentralWidget(split)

        QShortcut(QKeySequence.Save, self, activated=self._save_csv)
        opened = bool(self.config['plot_open'])
        self.plot_toggle.setChecked(opened)
        self._toggle_plot(opened)

    # ------------------------------------------------------------ キー操作

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if ev.type() != QEvent.KeyPress or not self.isActiveWindow() or self.bag is None:
            return False
        focus = QApplication.focusWidget()
        if isinstance(focus, (QAbstractSpinBox, QLineEdit)) or \
                (isinstance(focus, QComboBox) and focus.isEditable()):
            if ev.key() not in (Qt.Key_Return, Qt.Key_Enter):
                return False
        key, mod = ev.key(), ev.modifiers()
        step = 10 if mod & Qt.ShiftModifier else 1
        if key == Qt.Key_Left:
            self._set_frame(self.frame - step)
        elif key == Qt.Key_Right:
            self._set_frame(self.frame + step)
        elif key == Qt.Key_Home:
            self._set_frame(0)
        elif key == Qt.Key_End:
            self._set_frame(self._n_frames() - 1)
        elif key == Qt.Key_Space:
            self._toggle_play()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._add_interval()
        elif key == Qt.Key_E and not mod:
            self._set_end_here()
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_interval()
        else:
            return False
        return True

    # ------------------------------------------------------------ bag 読み込み

    def _scan_bags(self):
        cur = self.bag_dir
        self.bags = bc.find_bags(self.root)
        self.bag_list.blockSignals(True)
        self.bag_list.clear()
        for b in self.bags:
            mark = '✓ ' if bc.csv_path(b).exists() else '   '
            item = QListWidgetItem(mark + str(b.relative_to(self.root)))
            item.setToolTip(str(b))
            self.bag_list.addItem(item)
        if cur in self.bags:
            self.bag_list.setCurrentRow(self.bags.index(cur))
        self.bag_list.blockSignals(False)
        self.statusBar().showMessage(f'{len(self.bags)} 個の bag を検出 ({self.root})')

    def _on_bag_selected(self, row: int):
        if row < 0 or row >= len(self.bags) or self.bags[row] == self.bag_dir:
            return
        self._stop_play()
        path = self.bags[row]
        dlg = QProgressDialog(f'読み込み中: {path.name}', None, 0, 100, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(300)

        def progress(f):
            dlg.setValue(int(f * 100))
            QApplication.processEvents()
        try:
            bag = bc.load_bag(path, progress)
        except Exception as e:  # noqa: BLE001 - 読めない bag は通知だけして続行
            dlg.close()
            QMessageBox.warning(self, '読み込み失敗', f'{path}\n{e}')
            return
        dlg.close()
        self.bag, self.bag_dir = bag, path
        self.img_cache.clear()
        self.intervals, offset = bc.load_labels(path)
        self.video_offset = offset if offset is not None else float(self.config['video_offset'])
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(self.video_offset)
        self.offset_spin.blockSignals(False)

        def fill(combo, items, want):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if want in items:
                combo.setCurrentText(want)
            combo.blockSignals(False)
        fill(self.image_combo, list(bag.images), self.config['image_topic'])
        fill(self.ref_combo, list(bag.series), self.config['ref_topic'])
        joy_topics = [t for t, s in bag.series.items() if s.msgtype == 'sensor_msgs/msg/Joy']
        fill(self.joy_topic, joy_topics or [self.config['joy_topic']], self.config['joy_topic'])
        self.config['image_topic'] = self.image_combo.currentText()
        self.config['ref_topic'] = self.ref_combo.currentText() or self.config['ref_topic']
        self._build_tree()
        self._refresh_after_selection()
        self.plot_panel.t0, self.plot_panel.t1 = bag.start_ns, bag.end_ns
        self._update_plots()
        self.frame = 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(self._n_frames() - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.timeline.t0, self.timeline.t1 = bag.start_ns, bag.end_ns
        self._refresh_intervals()
        self._set_frame(0, force=True)
        n_img = self._n_frames()
        self.statusBar().showMessage(
            f'{path.name}: {(bag.end_ns - bag.start_ns) * 1e-9:.1f} s, 映像 {n_img} フレーム, '
            f'数値トピック {len(bag.series)} 個, 既存ラベル区間 {len(self.intervals)} 個')

    def _build_tree(self):
        self._building_tree = True
        self.tree.clear()
        sel = set(self.config['selected'])
        plots = set(self.config['plots'])
        for topic, s in self.bag.series.items():
            top = QTreeWidgetItem([topic, f'{s.msgtype}  ({len(s.times)} msgs)'])
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
            top.setCheckState(0, Qt.Unchecked)  # 子を追加する前に設定 (後だと子に伝播する)
            any_checked = False
            for fld in s.fields:
                child = QTreeWidgetItem([fld, ''])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                checked = f'{topic}|{fld}' in sel
                any_checked |= checked
                child.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                child.setData(0, Qt.UserRole, (topic, fld))
                plotted = f'{topic}|{fld}' in plots
                any_checked |= plotted
                child.setCheckState(1, Qt.Checked if plotted else Qt.Unchecked)
                top.addChild(child)
            self.tree.addTopLevelItem(top)
            top.setExpanded(any_checked)
        self._building_tree = False
        self._apply_filter(self.filter_edit.text())

    def _apply_filter(self, text: str):
        text = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            vis = 0
            for j in range(top.childCount()):
                c = top.child(j)
                hide = bool(text) and text not in f'{top.text(0)}/{c.text(0)}'.lower()
                c.setHidden(hide)
                vis += not hide
            top.setHidden(vis == 0)
            if text:
                top.setExpanded(True)

    def _on_tree_changed(self, item, col):
        if self._building_tree or self.bag is None:
            return
        if col == 1:
            if item.parent() is None:
                return
            key = '{}|{}'.format(*item.data(0, Qt.UserRole))
            plots = [k for k in self.config['plots'] if k != key]
            if item.checkState(1) == Qt.Checked:
                plots.append(key)  # チェックした順に上から並べる
            self._cfg('plots', plots)
            self._update_plots()
            return
        if col != 0:
            return
        # このbagに存在しないフィールドの選択は保持しつつ、表示中のものを更新
        present = {f'{t}|{f}' for t, s in self.bag.series.items() for f in s.fields}
        sel = [k for k in self.config['selected'] if k not in present]
        sel += [f'{t}|{f}' for t, f in self._columns()]
        self._cfg('selected', sel)
        self._refresh_after_selection()

    def _columns(self) -> list[tuple[str, str]]:
        cols = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                if c.checkState(0) == Qt.Checked:
                    cols.append(tuple(c.data(0, Qt.UserRole)))
        return cols

    def _refresh_after_selection(self):
        cols = self._columns()
        self.sel_label.setText(f'選択中: {len(cols)} 列')
        names = [bc.column_name(*c) for c in cols]
        want = self.config['plot_field']
        self.plot_combo.blockSignals(True)
        self.plot_combo.clear()
        self.plot_combo.addItems(names)
        if want in names:
            self.plot_combo.setCurrentText(want)
        self.plot_combo.blockSignals(False)
        self._update_wave()

    # ------------------------------------------------------------ 設定変更

    def _cfg(self, key, value):
        self.config[key] = value
        self._save_config()

    def _on_image_topic(self, topic):
        if topic:
            self._cfg('image_topic', topic)
            self.img_cache.clear()
            self.slider.setRange(0, max(self._n_frames() - 1, 0))
            self._set_frame(0, force=True)

    def _on_ref_topic(self, topic):
        if topic:
            self._cfg('ref_topic', topic)

    def _on_plot_field(self, name):
        if name:
            self._cfg('plot_field', name)
        self._update_wave()

    def _joy_changed(self, key, value):
        if key == 'joy_topic' and not value:
            return
        self._cfg(key, value)
        # 停止ボタン由来の終了時刻を再計算する
        for iv in self.intervals:
            if iv.stop_source == 'joy':
                self._recompute_stop(iv)
        self._refresh_intervals(save=self.bag is not None)

    def _update_wave(self):
        tl = self.timeline
        tl.wave_t, tl.wave_v, tl.wave_name = np.array([]), np.array([]), ''
        if self.bag is not None:
            for t, f in self._columns():
                if bc.column_name(t, f) == self.plot_combo.currentText():
                    s = self.bag.series[t]
                    tl.wave_t, tl.wave_v, tl.wave_name = s.times, s.fields[f], bc.column_name(t, f)
        tl.update()

    # ------------------------------------------------------------ プロット

    def _update_plots(self):
        sigs = []
        if self.bag is not None:
            for key in self.config['plots']:
                topic, _, fld = key.partition('|')
                s = self.bag.series.get(topic)
                if s is not None and fld in s.fields:
                    sigs.append((f'{topic}/{fld}', s.times, s.fields[fld]))
        self.plot_panel.set_signals(sigs)

    def _toggle_plot(self, opened: bool):
        self.plot_toggle.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)
        sizes = self.view_split.sizes()
        head = self.plot_toggle.sizeHint().height() + 4
        if not opened and sizes[1] > head:
            self._plot_sizes = sizes
        self.plot_panel.setVisible(opened)
        self.plot_window.setEnabled(opened)
        if opened:
            if self._plot_sizes:
                self.view_split.setSizes(self._plot_sizes)
        else:
            self.view_split.setSizes([sum(sizes) - head, head])
        self._cfg('plot_open', opened)
        self._show_current_image()

    def _on_plot_window(self, _):
        self.plot_panel.window = self.plot_window.currentData()
        self._cfg('plot_window', self.plot_panel.window)
        self.plot_panel.update()

    # ------------------------------------------------------------ 映像の表示調整

    def _on_view_adjust(self, _=None):
        vals = {}
        for key, (sl, val, fmt) in self.view_sliders.items():
            vals[key] = sl.value()
            val.setText(fmt(sl.value()))
            self.config[key] = sl.value()
        self._save_config()
        self.view_lut = adjust_lut(vals['view_brightness'], vals['view_contrast'],
                                   vals['view_gamma'])
        self._show_current_image()

    def _reset_view_adjust(self):
        for key, _, _, _, default, _ in VIEW_ADJUST:
            sl = self.view_sliders[key][0]
            sl.blockSignals(True)
            sl.setValue(default)
            sl.blockSignals(False)
        self._on_view_adjust()

    def _show_current_image(self):
        if self.bag is not None and self._n_frames():
            self._show_image(self.frame)

    # ------------------------------------------------------------ フレーム操作

    def _images(self) -> bc.ImageSeries | None:
        if self.bag is None:
            return None
        return self.bag.images.get(self.image_combo.currentText())

    def _n_frames(self) -> int:
        img = self._images()
        return len(img.times) if img else 0

    def _frame_time(self, i: int) -> int:
        img = self._images()
        return int(img.times[i]) - self._offset_ns() if img else self.bag.start_ns

    def _offset_ns(self) -> int:
        return int(round(self.video_offset * 1e9))

    def _seek_ns(self, t: int):
        img = self._images()
        if img is None:
            return
        t += self._offset_ns()
        i = int(np.searchsorted(img.times, t))
        if i > 0 and (i >= len(img.times) or t - img.times[i - 1] < img.times[i] - t):
            i -= 1
        self._set_frame(i)

    def _set_frame(self, i: int, force=False):
        n = self._n_frames()
        if self.bag is None:
            return
        if n == 0:
            self.video.setText('映像トピックがありません')
            self.video.setPixmap(QPixmap())
            return
        i = min(max(i, 0), n - 1)
        if i == self.frame and not force:
            return
        self.frame = i
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self._show_image(i)
        t = self._frame_time(i)
        self.timeline.cursor_ns = t
        self.timeline.update()
        self.plot_panel.cursor_ns = t
        self.plot_panel.update()
        self._update_info(t)

    def _show_image(self, i: int):
        rgb = self.img_cache.get(i)
        if rgb is None:
            arr = bc.decode_image(self._images(), i)
            if arr is None:
                self.video.setText('画像をデコードできません')
                return
            rgb = np.ascontiguousarray(arr[:, :, ::-1])
            if len(self.img_cache) > 200:
                self.img_cache.clear()
            self.img_cache[i] = rgb
        if self.view_lut is not None:
            rgb = self.view_lut[rgb]  # 表示用のコピーにだけ適用する
        h, w, _ = rgb.shape
        pm = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.bag is not None and self._n_frames():
            self._show_image(self.frame)

    def _value_at(self, topic, fld, t):
        s = self.bag.series.get(topic)
        if s is None or fld not in s.fields:
            return None
        k = int(np.searchsorted(s.times, t, side='right')) - 1
        return None if k < 0 else s.fields[fld][k]

    def _update_info(self, t: int):
        joy_key = f'{self.joy_kind.currentText()}[{self.joy_index.value()}]'
        jv = self._value_at(self.joy_topic.currentText(), joy_key, t)
        label = 0
        for iv in self.intervals:
            if iv.start_ns <= t < iv.stop_ns:
                label = iv.label
        parts = [f'frame {self.frame + 1}/{self._n_frames()}',
                 f'elapsed {(t - self.bag.start_ns) * 1e-9:.3f} s',
                 f'stamp {t // 1_000_000_000}.{t % 1_000_000_000:09d}',
                 f'{joy_key}={jv if jv is not None else "-"}',
                 f'label={label}',
                 f'offset {self.video_offset:+.3f} s']
        pf = self.plot_combo.currentText()
        for tp, f in self._columns():
            if bc.column_name(tp, f) == pf:
                v = self._value_at(tp, f, t)
                parts.append(f'{pf}={v if v is not None else "-"}')
        self.info.setText('   |   '.join(parts))

    # ------------------------------------------------------------ 再生

    def _toggle_play(self):
        if self.play_timer.isActive():
            self._stop_play()
        elif self._n_frames():
            if self.frame >= self._n_frames() - 1:
                self._set_frame(0)
            self.play_origin = (time.monotonic(), self._frame_time(self.frame))
            self.play_timer.start(30)
            self.btn_play.setText('⏸ 停止 (Space)')

    def _stop_play(self):
        self.play_timer.stop()
        self.btn_play.setText('▶ 再生 (Space)')

    def _play_tick(self):
        wall0, t0 = self.play_origin
        target = t0 + int((time.monotonic() - wall0) * 1e9)
        img = self._images()
        i = int(np.searchsorted(img.times, target + self._offset_ns(), side='right')) - 1
        self._set_frame(i)
        if i >= len(img.times) - 1:
            self._stop_play()

    # ------------------------------------------------------------ ラベル区間

    def _find_stop(self, source: str, start_ns: int) -> int | None:
        """start_ns より後の 'joy' (停止ボタン押下) / 'odom' (停止) の時刻."""
        if source == 'joy':
            key = f'{self.joy_kind.currentText()}[{self.joy_index.value()}]'
            return bc.find_stop(self.bag.series.get(self.joy_topic.currentText()), key,
                                self.joy_thr.value(), start_ns)
        if source == 'odom':
            after = self.timeline.stops[self.timeline.stops > start_ns]
            return int(after[0]) if after.size else None
        return None

    def _recompute_stop(self, iv: bc.Interval):
        """自動で決まる終了 (停止ボタン / odom / 未設定) を開始時刻から求め直す."""
        if iv.stop_source == 'manual':
            return
        stop = self._find_stop(iv.stop_source, iv.start_ns)
        if stop is None:
            iv.stop_ns, iv.stop_source = self.bag.end_ns + 1, 'bag_end'
        else:
            iv.stop_ns = stop

    def _add_interval(self):
        if self.bag is None or not self._n_frames():
            return
        start = self._frame_time(self.frame)
        mode = self.end_mode.currentData()
        if mode == 'odom_then_after':
            self._add_contact_and_after(start)
            return
        iv = bc.Interval(start, start, self.label_spin.value(),
                         'bag_end' if mode == 'manual' else mode)
        self._recompute_stop(iv)
        self.intervals.append(iv)
        self._refresh_intervals(save=True, select=len(self.intervals) - 1)
        if mode == 'manual':
            self.statusBar().showMessage(
                '区間を開始しました。終わりのフレームで E を押してください '
                '(未設定のままなら bag 末尾まで label になります)')
        elif iv.stop_source == 'bag_end':
            self.statusBar().showMessage(
                f'{STOP_SOURCE_NAMES[mode]} が見つからないため bag の末尾までを区間にしました '
                '(E キーで終了位置を変更できます)')
        else:
            self.statusBar().showMessage(
                f'区間追加: {(iv.stop_ns - start) * 1e-9:.3f} s ({STOP_SOURCE_NAMES[mode]}で終了)')

    def _add_contact_and_after(self, start: int):
        """start から odom 停止までを label、停止から bag 末尾までを AFTER_STOP_LABEL にする."""
        iv = bc.Interval(start, start, self.label_spin.value(), 'odom')
        self._recompute_stop(iv)
        self.intervals.append(iv)
        if iv.stop_source == 'bag_end':
            self._refresh_intervals(save=True, select=len(self.intervals) - 1)
            self.statusBar().showMessage(
                'この位置より後に odom の停止が見つからないため bag の末尾までを区間にしました '
                '(E キーで終了位置を変更できます)')
            return
        # 区間は [start, stop) なので、停止時刻以降の行が label 2 になり重ならない
        self.intervals.append(bc.Interval(iv.stop_ns, self.bag.end_ns + 1, AFTER_STOP_LABEL,
                                          'bag_end', start_source='after_prev'))
        self._refresh_intervals(save=True, select=len(self.intervals) - 2)
        self.statusBar().showMessage(
            f'label {iv.label}: {(iv.stop_ns - start) * 1e-9:.3f} s (odom 停止まで), '
            f'以降 bag 末尾まで label {AFTER_STOP_LABEL}')

    def _link_intervals(self):
        """開始が直前の区間の終了に連動する区間を更新する."""
        for prev, iv in zip(self.intervals, self.intervals[1:]):
            if iv.start_source == 'after_prev':
                iv.start_ns = prev.stop_ns

    def _set_end_here(self):
        row = self.interval_list.currentRow()
        if row < 0:
            row = len(self.intervals) - 1
        if row < 0 or self.bag is None:
            self.statusBar().showMessage('先に Enter で区間を開始してください')
            return
        iv = self.intervals[row]
        t = self._frame_time(self.frame)
        if t <= iv.start_ns:
            self.statusBar().showMessage('終了は開始より後にしてください')
            return
        iv.stop_ns, iv.stop_source = t, 'manual'
        self._refresh_intervals(save=True, select=row)

    def _delete_interval(self):
        row = self.interval_list.currentRow()
        if row < 0:
            row = len(self.intervals) - 1
        if row < 0:
            return
        del self.intervals[row]
        # 消した区間に連動していた区間は、今の開始位置のまま独立させる
        if row < len(self.intervals) and self.intervals[row].start_source == 'after_prev':
            self.intervals[row].start_source = 'manual'
        self._refresh_intervals(save=True, select=min(row, len(self.intervals) - 1))

    def _on_interval_selected(self, row):
        self.timeline.selected = row
        self.timeline.update()

    def _refresh_intervals(self, save=False, select=None):
        if self.bag is None:
            return
        self._link_intervals()
        self.interval_list.blockSignals(True)
        self.interval_list.clear()
        for k, iv in enumerate(self.intervals):
            s = (iv.start_ns - self.bag.start_ns) * 1e-9
            e = (min(iv.stop_ns, self.bag.end_ns) - self.bag.start_ns) * 1e-9
            linked = f'  開始: #{k} の終了に連動' if iv.start_source == 'after_prev' else ''
            self.interval_list.addItem(
                f'#{k + 1}  {s:8.3f} s → {e:8.3f} s  ({e - s:.3f} s)  '
                f'label={iv.label}  終了: {STOP_SOURCE_NAMES.get(iv.stop_source, iv.stop_source)}'
                + linked)
        if select is not None:
            self.interval_list.setCurrentRow(select)
        self.interval_list.blockSignals(False)
        joy_key = f'{self.joy_kind.currentText()}[{self.joy_index.value()}]'
        js = self.bag.series.get(self.joy_topic.currentText())
        self.timeline.edges = bc.rising_edges(js, joy_key, self.joy_thr.value()) \
            if js else np.array([])
        self.timeline.stops = bc.stop_events(self.bag.series.get(SPEED_TOPIC), SPEED_FIELDS)
        self.timeline.intervals = self.intervals
        self.timeline.selected = self.interval_list.currentRow()
        self.timeline.update()
        self.plot_panel.intervals = self.intervals
        self.plot_panel.edges = self.timeline.edges
        self.plot_panel.stops = self.timeline.stops
        self.plot_panel.update()
        if save:
            self._save_labels()
        self._update_info(self.timeline.cursor_ns)

    # ------------------------------------------------------------ 映像オフセット

    def _save_labels(self):
        bc.save_labels(self.bag_dir, self.intervals, self.video_offset)

    def _set_offset(self, value: float):
        """映像オフセットを変更する. 区間はフレームを見て決めたものなので一緒にずらす."""
        if self.bag is None:
            return
        delta = int(round((value - self.video_offset) * 1e9))
        self.video_offset = value
        self._cfg('video_offset', value)  # 次に開く bag の初期値
        for iv in self.intervals:
            iv.start_ns -= delta
            if iv.stop_source == 'manual':
                iv.stop_ns -= delta
            else:
                self._recompute_stop(iv)
        self._refresh_intervals(save=True)
        self._set_frame(self.frame, force=True)

    def _sync_offset(self):
        """現在フレームを「映像でロボットが止まった瞬間」とみなしてオフセットを決める."""
        img = self._images()
        stops = self.timeline.stops
        if img is None or not len(stops):
            self.statusBar().showMessage(f'{SPEED_TOPIC} から停止時刻が見つからないため同期できません')
            return
        raw = int(img.times[self.frame])
        # 映像は遅れて記録されるので、このフレームより前の停止のうち最も近いものを選ぶ
        before = stops[stops <= raw]
        stop = int(before[-1]) if before.size else int(stops[0])
        self.offset_spin.setValue((raw - stop) * 1e-9)
        self.statusBar().showMessage(
            f'映像オフセットを {self.video_offset:.3f} s に設定しました '
            f'(odom 停止 {(stop - self.bag.start_ns) * 1e-9:.3f} s に合わせた)')

    # ------------------------------------------------------------ 保存

    def _save_csv(self):
        if self.bag is None:
            return
        cols = self._columns()
        ref = self.ref_combo.currentText()
        if not cols:
            QMessageBox.warning(self, '保存できません', 'データ軸にするフィールドを選択してください')
            return
        if ref not in self.bag.series:
            QMessageBox.warning(self, '保存できません', '基準トピックを選択してください')
            return
        if not self.intervals and QMessageBox.question(
                self, '確認', 'ラベル区間がありません (全行 label=0)。保存しますか?') \
                != QMessageBox.Yes:
            return
        header, t, values, labels = bc.build_dataset(self.bag, ref, cols, self.intervals)
        out = bc.csv_path(self.bag_dir)
        bc.write_csv(out, self.bag, header, t, values, labels)
        self._save_labels()
        n1 = int((labels != 0).sum())
        self._scan_bags()
        self.statusBar().showMessage(
            f'保存しました: {out}  ({len(t)} 行 × {len(header)} 列, label≠0: {n1} 行)')


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
    app = QApplication(sys.argv)
    win = MainWindow(root)
    win.show()
    # Ctrl+C で KeyboardInterrupt を Qt のコールバック内に投げると abort するので、
    # 代わりにイベントループを終了させる。ラベル区間は変更のたびに保存済み。
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    # Qt のイベントループ中も定期的に Python に制御を戻し、シグナルハンドラを実行させる
    tick = QTimer()
    tick.timeout.connect(lambda: None)
    tick.start(200)
    code = app.exec_()
    print('終了しました')
    sys.exit(code)


if __name__ == '__main__':
    main()
