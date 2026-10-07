"""Main desktop window."""
from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..core import config, imaging
from ..core.engine import RESOLUTIONS, SAMPLERS, Engine, Job, LoraSpec
from ..core.presets import load_presets
from .widgets import ImageView, MaskCanvas, pil_to_qimage
from .worker import Task

MODE_LABELS = {
    "txt2img": "Текст → изображение",
    "img2img": "Изображение → изображение",
    "inpaint": "Inpaint (перерисовка по маске)",
    "outpaint": "Outpaint (расширение с дорисовкой)",
}
VRAM_LABELS = {
    "balanced": "Экономия VRAM (рекомендуется для 8 ГБ)",
    "fast": "Всё в видеопамяти (быстрее, если влезает)",
}


def _spin(lo, hi, val, step=1, decimals=None):
    w = QDoubleSpinBox() if decimals is not None else QSpinBox()
    if decimals is not None:
        w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(val)
    return w


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SemiReal Studio")
        self.resize(1500, 950)
        self.settings = config.Settings.load()
        self.engine = Engine(self.settings)
        self.presets = load_presets()
        self.task: Task | None = None
        self.last_seed: int | None = None
        self.pose_source: Image.Image | None = None
        self.results: list[tuple[Path, str]] = []

        splitter = QSplitter()
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_views())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 980])
        self.setCentralWidget(splitter)

        self._refresh_lists()
        self._restore_ui()
        self._on_mode_changed()
        if not torch_cuda_ok():
            self.status.setText("⚠ CUDA не найдена — генерация пойдёт на процессоре (очень медленно). См. README.")

    # ================================================================ layout

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)

        # ---- model
        g = QGroupBox("Модель")
        f = QFormLayout(g)
        row = QHBoxLayout()
        self.checkpoint = QComboBox()
        self.checkpoint.setMinimumWidth(250)
        refresh = QPushButton("⟳")
        refresh.setToolTip("Обновить списки моделей, LoRA, апскейлеров и поз")
        refresh.setFixedWidth(32)
        refresh.clicked.connect(self._refresh_lists)
        row.addWidget(self.checkpoint, 1)
        row.addWidget(refresh)
        f.addRow("Чекпоинт:", row)
        self.vram = QComboBox()
        for k, v in VRAM_LABELS.items():
            self.vram.addItem(v, k)
        self.vram.setCurrentIndex(max(0, self.vram.findData(self.settings.vram_mode)))
        self.vram.currentIndexChanged.connect(self._on_engine_option)
        f.addRow("Память:", self.vram)
        self.vae_fix = QCheckBox("VAE fp16-fix (защита от чёрных картинок)")
        self.vae_fix.setChecked(self.settings.vae_fp16_fix)
        self.vae_fix.toggled.connect(self._on_engine_option)
        f.addRow(self.vae_fix)
        lay.addWidget(g)

        tabs = QTabWidget()
        tabs.addTab(self._tab_main(), "Основное")
        tabs.addTab(self._tab_extras(), "LoRA и поза")
        tabs.addTab(self._tab_quality(), "Качество")
        lay.addWidget(tabs, 1)

        # ---- run
        row = QHBoxLayout()
        self.btn_generate = QPushButton("Сгенерировать  (Ctrl+Enter)")
        self.btn_generate.setMinimumHeight(40)
        self.btn_generate.setStyleSheet("QPushButton { font-weight: bold; }")
        self.btn_generate.clicked.connect(self.generate)
        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        row.addWidget(self.btn_generate, 3)
        row.addWidget(self.btn_stop, 1)
        lay.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        lay.addWidget(self.progress)
        self.status = QLabel("Готово.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        act = QAction(self)
        act.setShortcuts([QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")])
        act.triggered.connect(self.generate)
        self.addAction(act)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(460)
        return scroll

    def _tab_main(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # ---- mode
        g = QGroupBox("Режим")
        f = QFormLayout(g)
        self.mode = QComboBox()
        for k, v in MODE_LABELS.items():
            self.mode.addItem(v, k)
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        f.addRow(self.mode)
        self.btn_load_init = QPushButton("Открыть исходное изображение…")
        self.btn_load_init.clicked.connect(self._open_init)
        f.addRow(self.btn_load_init)
        self.strength = _spin(0.05, 1.0, 0.6, 0.05, 2)
        self.strength.setToolTip("Насколько сильно менять исходник: 0.3 — слегка, 0.7 — сильно, 1.0 — с нуля")
        self.lbl_strength = QLabel("Сила изменения:")
        f.addRow(self.lbl_strength, self.strength)
        self.mask_blur = _spin(0, 64, 8)
        self.lbl_mask_blur = QLabel("Размытие маски:")
        f.addRow(self.lbl_mask_blur, self.mask_blur)
        self.only_masked = QCheckBox("Перерисовывать только область маски (больше деталей)")
        f.addRow(self.only_masked)
        self.out_box = QWidget()
        ol = QHBoxLayout(self.out_box)
        ol.setContentsMargins(0, 0, 0, 0)
        self.out_sides = {}
        for key, label in (("left", "←"), ("right", "→"), ("top", "↑"), ("bottom", "↓")):
            s = _spin(0, 1024, 0, 64)
            s.setPrefix(label + " ")
            s.setSuffix(" px")
            self.out_sides[key] = s
            ol.addWidget(s)
        self.lbl_out = QLabel("Расширить:")
        f.addRow(self.lbl_out, self.out_box)
        lay.addWidget(g)

        # ---- prompt
        g = QGroupBox("Промпт")
        v = QVBoxLayout(g)
        self.preset = QComboBox()
        for p in self.presets:
            self.preset.addItem(p.name)
        self.preset.currentIndexChanged.connect(self._on_preset)
        self.preset_note = QLabel()
        self.preset_note.setWordWrap(True)
        self.preset_note.setStyleSheet("color: #888;")
        v.addWidget(QLabel("Стиль (пресет):"))
        v.addWidget(self.preset)
        v.addWidget(self.preset_note)
        v.addWidget(QLabel("Что нарисовать (англ., поддерживается (слово:1.2)):"))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("1woman, adult, long red hair, ... ")
        self.prompt.setMinimumHeight(110)
        v.addWidget(self.prompt)
        v.addWidget(QLabel("Негативный промпт (дополняет негатив пресета):"))
        self.negative = QPlainTextEdit()
        self.negative.setMaximumHeight(70)
        v.addWidget(self.negative)
        lay.addWidget(g)

        # ---- params
        g = QGroupBox("Параметры")
        f = QFormLayout(g)
        self.resolution = QComboBox()
        for wv, hv in RESOLUTIONS:
            self.resolution.addItem(f"{wv} × {hv}", (wv, hv))
        self.resolution.addItem("Свой размер", None)
        self.resolution.currentIndexChanged.connect(self._on_resolution)
        f.addRow("Размер:", self.resolution)
        row = QHBoxLayout()
        self.width_ = _spin(512, 2048, 832, 64)
        self.height_ = _spin(512, 2048, 1216, 64)
        row.addWidget(self.width_)
        row.addWidget(QLabel("×"))
        row.addWidget(self.height_)
        f.addRow("", row)
        self.steps = _spin(5, 100, 28)
        f.addRow("Шаги:", self.steps)
        self.cfg = _spin(1.0, 20.0, 6.0, 0.5, 1)
        f.addRow("CFG:", self.cfg)
        self.sampler = QComboBox()
        self.sampler.addItems(list(SAMPLERS))
        f.addRow("Сэмплер:", self.sampler)
        self.clip_skip = _spin(1, 4, 2)
        f.addRow("Clip skip:", self.clip_skip)
        row = QHBoxLayout()
        self.seed = _spin(-1, 2**31 - 1, -1)
        self.seed.setSpecialValueText("случайный")
        b_rand = QPushButton("🎲")
        b_rand.setFixedWidth(32)
        b_rand.setToolTip("Случайный сид")
        b_rand.clicked.connect(lambda: self.seed.setValue(-1))
        b_last = QPushButton("♻")
        b_last.setFixedWidth(32)
        b_last.setToolTip("Повторить сид последней картинки")
        b_last.clicked.connect(lambda: self.last_seed is not None and self.seed.setValue(self.last_seed))
        row.addWidget(self.seed, 1)
        row.addWidget(b_rand)
        row.addWidget(b_last)
        f.addRow("Сид:", row)
        self.batch = _spin(1, 16, 1)
        f.addRow("Количество:", self.batch)
        lay.addWidget(g)
        lay.addStretch()
        return w

    def _tab_extras(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        g = QGroupBox("LoRA")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.lora_pick = QComboBox()
        self.lora_pick.setMinimumWidth(200)
        b_add = QPushButton("Добавить")
        b_add.clicked.connect(lambda: self._add_lora(self.lora_pick.currentText(), 0.8))
        row.addWidget(self.lora_pick, 1)
        row.addWidget(b_add)
        v.addLayout(row)
        self.lora_table = QTableWidget(0, 3)
        self.lora_table.setHorizontalHeaderLabels(["Файл", "Вес", ""])
        self.lora_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.lora_table.verticalHeader().setVisible(False)
        self.lora_table.setMinimumHeight(140)
        v.addWidget(self.lora_table)
        hint = QLabel("Файлы .safetensors кладите в models/loras. Не забудьте триггер-слова LoRA в промпте.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        v.addWidget(hint)
        lay.addWidget(g)

        g = QGroupBox("Поза (OpenPose)")
        f = QFormLayout(g)
        self.pose_on = QCheckBox("Использовать позу (только «Текст → изображение»)")
        self.pose_on.toggled.connect(lambda _: self._update_pose_preview())
        f.addRow(self.pose_on)
        row = QHBoxLayout()
        self.pose_lib = QComboBox()
        self.pose_lib.currentIndexChanged.connect(self._on_pose_lib)
        b_file = QPushButton("Файл…")
        b_file.clicked.connect(self._open_pose)
        row.addWidget(self.pose_lib, 1)
        row.addWidget(b_file)
        f.addRow("Поза:", row)
        self.pose_extract = QCheckBox("Это обычное фото — извлечь из него скелет")
        f.addRow(self.pose_extract)
        self.pose_strength = _spin(0.1, 1.5, 0.9, 0.05, 2)
        f.addRow("Сила позы:", self.pose_strength)
        self.pose_preview = QLabel("нет позы")
        self.pose_preview.setAlignment(Qt.AlignCenter)
        self.pose_preview.setFixedHeight(200)
        self.pose_preview.setStyleSheet("background: #1e1e1e; color: #888;")
        f.addRow(self.pose_preview)
        hint = QLabel("Готовые скелеты (цветные палочки на чёрном) кладите в models/poses. "
                      "Из фото скелет извлекается автоматически при генерации.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        f.addRow(hint)
        lay.addWidget(g)
        lay.addStretch()
        return w

    def _tab_quality(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        g = QGroupBox("Апскейл (Hires fix)")
        f = QFormLayout(g)
        self.hires = QCheckBox("Увеличить и дорисовать детали")
        f.addRow(self.hires)
        self.hires_scale = _spin(1.1, 2.0, 1.5, 0.1, 2)
        f.addRow("Масштаб:", self.hires_scale)
        self.upscaler = QComboBox()
        f.addRow("Апскейлер:", self.upscaler)
        self.hires_denoise = _spin(0.1, 0.8, 0.35, 0.05, 2)
        f.addRow("Сила:", self.hires_denoise)
        self.hires_steps = _spin(5, 50, 15)
        f.addRow("Шаги:", self.hires_steps)
        lay.addWidget(g)

        g = QGroupBox("Исправление лиц и рук (как ADetailer)")
        f = QFormLayout(g)
        self.fix_faces = QCheckBox("Лица")
        self.fix_hands = QCheckBox("Руки")
        row = QHBoxLayout()
        row.addWidget(self.fix_faces)
        row.addWidget(self.fix_hands)
        f.addRow(row)
        self.detail_strength = _spin(0.1, 0.8, 0.4, 0.05, 2)
        f.addRow("Сила:", self.detail_strength)
        self.detail_prompt = QLineEdit()
        self.detail_prompt.setPlaceholderText("пусто = основной промпт")
        f.addRow("Промпт:", self.detail_prompt)
        lay.addWidget(g)

        info = QLabel(
            "Ориентир по времени на RTX 4070 Laptop 8 ГБ: 832×1216, 28 шагов ≈ 20–40 с; "
            "+ Hires ×1.5 ≈ +60–90 с; + лица/руки ≈ +15 с на каждую область."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888;")
        lay.addWidget(info)
        lay.addStretch()
        return w

    def _build_views(self) -> QWidget:
        self.views = QTabWidget()

        res = QWidget()
        v = QVBoxLayout(res)
        self.result_view = ImageView("Здесь появится результат")
        v.addWidget(self.result_view, 1)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setFlow(QListWidget.LeftToRight)
        self.gallery.setWrapping(False)
        self.gallery.setIconSize(QSize(96, 96))
        self.gallery.setFixedHeight(124)
        self.gallery.currentRowChanged.connect(self._show_result)
        v.addWidget(self.gallery)
        row = QHBoxLayout()
        for text, mode in (("→ Img2img", "img2img"), ("→ Inpaint", "inpaint"), ("→ Outpaint", "outpaint")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, m=mode: self._send_to(m))
            row.addWidget(b)
        b = QPushButton("Взять сид")
        b.clicked.connect(self._reuse_seed)
        row.addWidget(b)
        b = QPushButton("Папка с результатами")
        b.clicked.connect(self._open_outputs)
        row.addWidget(b)
        v.addLayout(row)
        self.params_view = QPlainTextEdit()
        self.params_view.setReadOnly(True)
        self.params_view.setMaximumHeight(90)
        v.addWidget(self.params_view)
        self.views.addTab(res, "Результат")

        src = QWidget()
        v = QVBoxLayout(src)
        row = QHBoxLayout()
        b = QPushButton("Открыть…")
        b.clicked.connect(self._open_init)
        row.addWidget(b)
        self.btn_clear_mask = QPushButton("Очистить маску")
        self.btn_invert_mask = QPushButton("Инвертировать")
        row.addWidget(self.btn_clear_mask)
        row.addWidget(self.btn_invert_mask)
        self.mask_hint = QLabel("ЛКМ — рисовать маску, ПКМ — стирать, колесо — размер кисти")
        self.mask_hint.setStyleSheet("color: #888;")
        row.addWidget(self.mask_hint, 1)
        v.addLayout(row)
        self.canvas = MaskCanvas()
        self.btn_clear_mask.clicked.connect(self.canvas.clear_mask)
        self.btn_invert_mask.clicked.connect(self.canvas.invert_mask)
        v.addWidget(self.canvas, 1)
        self.views.addTab(src, "Исходник / маска")
        return self.views

    # ================================================================ state

    def _refresh_lists(self) -> None:
        def refill(combo: QComboBox, items: list[str], empty: str | None = None):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if empty:
                combo.addItem(empty)
            combo.addItems(items)
            idx = combo.findText(cur)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

        refill(self.checkpoint, config.list_checkpoints())
        if self.settings.last_checkpoint and self.checkpoint.findText(self.settings.last_checkpoint) >= 0 \
                and not self.checkpoint.property("touched"):
            self.checkpoint.setCurrentText(self.settings.last_checkpoint)
            self.checkpoint.setProperty("touched", True)
        refill(self.lora_pick, config.list_files(config.LORAS_DIR, config.LORA_EXT))
        refill(self.upscaler, imaging.Upscaler.available())
        refill(self.pose_lib, config.list_files(config.POSES_DIR, config.POSE_EXT), "— выберите —")
        if self.checkpoint.count() == 0:
            self.status.setText("Положите SDXL-чекпоинт (.safetensors) в папку models/checkpoints и нажмите ⟳.")

    def _restore_ui(self) -> None:
        ui = self.settings.ui
        if not ui:
            self._on_preset()
            return
        self.preset.blockSignals(True)
        self.preset.setCurrentText(ui.get("preset", ""))
        self.preset.blockSignals(False)
        self.preset_note.setText(self.presets[self.preset.currentIndex()].note)
        self.prompt.setPlainText(ui.get("prompt", ""))
        self.negative.setPlainText(ui.get("negative", ""))
        for name in ("steps", "cfg", "clip_skip", "batch", "width_", "height_", "hires_scale", "hires_denoise",
                     "hires_steps", "detail_strength", "pose_strength", "strength", "mask_blur"):
            if name in ui:
                getattr(self, name).setValue(ui[name])
        for name in ("hires", "fix_faces", "fix_hands", "only_masked"):
            getattr(self, name).setChecked(bool(ui.get(name, False)))
        self.sampler.setCurrentText(ui.get("sampler", self.sampler.currentText()))
        self.upscaler.setCurrentText(ui.get("upscaler", self.upscaler.currentText()))
        self.detail_prompt.setText(ui.get("detail_prompt", ""))
        idx = self.resolution.findData(tuple(ui.get("resolution") or ()) or None)
        self.resolution.setCurrentIndex(idx if idx >= 0 else 0)
        for spec in ui.get("loras", []):
            self._add_lora(spec[0], spec[1])

    def _save_ui(self) -> None:
        self.settings.ui = {
            "preset": self.preset.currentText(), "prompt": self.prompt.toPlainText(),
            "negative": self.negative.toPlainText(), "sampler": self.sampler.currentText(),
            "upscaler": self.upscaler.currentText(), "detail_prompt": self.detail_prompt.text(),
            "resolution": self.resolution.currentData(), "loras": [[s.file, s.weight] for s in self._loras()],
            **{n: getattr(self, n).value() for n in (
                "steps", "cfg", "clip_skip", "batch", "width_", "height_", "hires_scale", "hires_denoise",
                "hires_steps", "detail_strength", "pose_strength", "strength", "mask_blur")},
            **{n: getattr(self, n).isChecked() for n in ("hires", "fix_faces", "fix_hands", "only_masked")},
        }
        self.settings.last_checkpoint = self.checkpoint.currentText()
        self.settings.save()

    def closeEvent(self, e):  # noqa: N802
        if self.task and self.task.isRunning():
            self.task.cancel()
            self.task.wait(10000)
        self._save_ui()
        super().closeEvent(e)

    # ================================================================ handlers

    def _on_engine_option(self) -> None:
        self.settings.vram_mode = self.vram.currentData()
        self.settings.vae_fp16_fix = self.vae_fix.isChecked()
        if self.engine.base is not None and not (self.task and self.task.isRunning()):
            self.engine.unload()
            self.status.setText("Настройки памяти изменены — модель перезагрузится при следующей генерации.")

    def _on_mode_changed(self) -> None:
        mode = self.mode.currentData()
        source = mode != "txt2img"
        self.btn_load_init.setVisible(source)
        for wdg in (self.strength, self.lbl_strength):
            wdg.setVisible(source)
        for wdg in (self.mask_blur, self.lbl_mask_blur, self.only_masked):
            wdg.setVisible(mode == "inpaint")
        for wdg in (self.out_box, self.lbl_out):
            wdg.setVisible(mode == "outpaint")
        self.canvas.mask_enabled = mode == "inpaint"
        for wdg in (self.btn_clear_mask, self.btn_invert_mask, self.mask_hint):
            wdg.setVisible(mode == "inpaint")
        self.canvas.update()
        if mode == "outpaint" and self.strength.value() < 0.9:
            self.strength.setValue(1.0)
        if source:
            self.views.setCurrentIndex(1)

    def _on_preset(self) -> None:
        p = self.presets[self.preset.currentIndex()]
        self.preset_note.setText(p.note)
        s = p.settings
        if "sampler" in s:
            self.sampler.setCurrentText(s["sampler"])
        for key, widget in (("steps", self.steps), ("cfg", self.cfg), ("clip_skip", self.clip_skip)):
            if key in s:
                widget.setValue(s[key])

    def _on_resolution(self) -> None:
        data = self.resolution.currentData()
        custom = data is None
        self.width_.setEnabled(custom)
        self.height_.setEnabled(custom)
        if data:
            self.width_.setValue(data[0])
            self.height_.setValue(data[1])

    def _open_image(self, title: str) -> Image.Image | None:
        path, _ = QFileDialog.getOpenFileName(self, title, str(config.OUTPUTS_DIR),
                                              "Изображения (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path:
            return None
        try:
            img = Image.open(path)
            img.load()
            return img.convert("RGB")
        except OSError as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть файл: {e}")
            return None

    def _open_init(self) -> None:
        img = self._open_image("Исходное изображение")
        if img is not None:
            self.canvas.set_image(img)
            self.views.setCurrentIndex(1)
            if self.mode.currentData() == "txt2img":
                self.mode.setCurrentIndex(self.mode.findData("img2img"))

    def _open_pose(self) -> None:
        img = self._open_image("Поза: скелет OpenPose или фото")
        if img is not None:
            self.pose_source = img
            self.pose_lib.blockSignals(True)
            self.pose_lib.setCurrentIndex(0)
            self.pose_lib.blockSignals(False)
            self.pose_on.setChecked(True)
            self._update_pose_preview()

    def _on_pose_lib(self) -> None:
        if self.pose_lib.currentIndex() <= 0:
            return
        try:
            self.pose_source = Image.open(config.POSES_DIR / self.pose_lib.currentText()).convert("RGB")
        except OSError:
            self.pose_source = None
        self.pose_extract.setChecked(False)
        self.pose_on.setChecked(True)
        self._update_pose_preview()

    def _update_pose_preview(self, img: Image.Image | None = None) -> None:
        img = img or self.pose_source
        if img is None or not self.pose_on.isChecked():
            self.pose_preview.setPixmap(QPixmap())
            self.pose_preview.setText("нет позы" if img is None else "поза выключена")
            return
        pm = QPixmap.fromImage(pil_to_qimage(img))
        self.pose_preview.setPixmap(pm.scaled(self.pose_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _add_lora(self, file: str, weight: float) -> None:
        if not file or any(s.file == file for s in self._loras()):
            return
        r = self.lora_table.rowCount()
        self.lora_table.insertRow(r)
        item = QTableWidgetItem(file)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.lora_table.setItem(r, 0, item)
        self.lora_table.setCellWidget(r, 1, _spin(-2.0, 2.0, weight, 0.05, 2))
        rm = QPushButton("✕")
        rm.clicked.connect(lambda: self._remove_lora(item))
        self.lora_table.setCellWidget(r, 2, rm)
        self.lora_table.resizeColumnToContents(2)

    def _remove_lora(self, item) -> None:
        self.lora_table.removeRow(self.lora_table.row(item))

    def _loras(self) -> list[LoraSpec]:
        out = []
        for r in range(self.lora_table.rowCount()):
            out.append(LoraSpec(self.lora_table.item(r, 0).text(), self.lora_table.cellWidget(r, 1).value()))
        return out

    # ================================================================ generation

    def _build_job(self) -> Job:
        preset = self.presets[self.preset.currentIndex()]
        prompt, negative = preset.build(self.prompt.toPlainText(), self.negative.toPlainText())
        mode = self.mode.currentData()
        job = Job(
            mode=mode, prompt=prompt, negative=negative,
            width=self.width_.value(), height=self.height_.value(), steps=self.steps.value(),
            cfg=self.cfg.value(), sampler=self.sampler.currentText(), clip_skip=self.clip_skip.value(),
            seed=self.seed.value(), batch=self.batch.value(), loras=self._loras(),
            strength=self.strength.value(), mask_blur=self.mask_blur.value(),
            inpaint_only_masked=self.only_masked.isChecked(),
            outpaint=tuple(self.out_sides[k].value() for k in ("left", "right", "top", "bottom")),
            pose_strength=self.pose_strength.value(),
            hires=self.hires.isChecked(), hires_scale=self.hires_scale.value(),
            hires_denoise=self.hires_denoise.value(), hires_steps=self.hires_steps.value(),
            upscaler=self.upscaler.currentText(),
            fix_faces=self.fix_faces.isChecked(), fix_hands=self.fix_hands.isChecked(),
            detail_strength=self.detail_strength.value(), detail_prompt=self.detail_prompt.text(),
        )
        if mode != "txt2img":
            job.init_image = self.canvas.image()
            job.mask = self.canvas.mask() if mode == "inpaint" else None
        elif self.pose_on.isChecked() and self.pose_source is not None:
            job.pose_image = self.pose_source
        return job

    def generate(self) -> None:
        if self.task and self.task.isRunning():
            return
        checkpoint = self.checkpoint.currentText()
        if not checkpoint:
            QMessageBox.information(self, "Нет модели",
                                    "Положите SDXL-чекпоинт (.safetensors) в models/checkpoints и нажмите ⟳.")
            return
        job = self._build_job()
        if job.seed < 0:  # pick it here so "♻" can reuse it
            job.seed = random.randint(0, 2**31 - 1)
        extract = job.pose_image is not None and self.pose_extract.isChecked()
        self._save_ui()

        def work(task: Task):
            self.engine.load(checkpoint, task.progress.emit)
            if extract:
                task.progress.emit("Извлечение позы из фото…", 0, 0)
                job.pose_image = imaging.extract_pose(job.pose_image)
                task.pose.emit(job.pose_image)
            return self.engine.generate(job, task.progress.emit, task.cancelled,
                                        lambda img, path, params: task.image.emit(img, str(path), params))

        self.last_seed = job.seed
        self.task = Task(work)
        self.task.progress.connect(self._on_progress)
        self.task.image.connect(self._on_image)
        self.task.pose.connect(self._update_pose_preview)
        self.task.done.connect(self._on_done)
        self.task.failed.connect(self._on_failed)
        self.task.finished.connect(lambda: self._set_busy(False))
        self._set_busy(True)
        self.task.start()

    def stop(self) -> None:
        if self.task:
            self.task.cancel()
            self.status.setText("Останавливаю…")

    def _set_busy(self, busy: bool) -> None:
        self.btn_generate.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        if not busy:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

    def _on_progress(self, stage: str, step: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)  # busy indicator
            self.progress.setFormat("")
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(step)
            self.progress.setFormat(f"{stage}: {step}/{total}")
        self.status.setText(stage)

    def _on_image(self, img: Image.Image, path: str, params: str) -> None:
        self.results.append((Path(path), params))
        thumb = img.copy()
        thumb.thumbnail((192, 192))
        item = QListWidgetItem(QIcon(QPixmap.fromImage(pil_to_qimage(thumb))), "")
        item.setToolTip(path)
        self.gallery.addItem(item)
        self.gallery.setCurrentRow(self.gallery.count() - 1)
        self.views.setCurrentIndex(0)

    def _show_result(self, row: int) -> None:
        if not 0 <= row < len(self.results):
            return
        path, params = self.results[row]
        try:
            self.result_view.set_image(Image.open(path))
        except OSError:
            self.result_view.set_image(None)
        self.params_view.setPlainText(params)

    def _on_done(self, paths) -> None:
        self.status.setText(f"Готово: сохранено {len(paths)} шт. в {config.OUTPUTS_DIR}")

    def _on_failed(self, message: str) -> None:
        self.status.setText(message)
        if message != "Остановлено.":
            QMessageBox.warning(self, "Ошибка", message)

    # ================================================================ result actions

    def _current_result(self) -> tuple[Path, str] | None:
        row = self.gallery.currentRow()
        return self.results[row] if 0 <= row < len(self.results) else None

    def _send_to(self, mode: str) -> None:
        cur = self._current_result()
        if not cur:
            return
        self.canvas.set_image(Image.open(cur[0]).convert("RGB"))
        self.mode.setCurrentIndex(self.mode.findData(mode))
        self.views.setCurrentIndex(1)

    def _reuse_seed(self) -> None:
        cur = self._current_result()
        if not cur:
            return
        for part in cur[1].splitlines()[-1].split(", "):
            if part.startswith("Seed: "):
                self.seed.setValue(int(part[6:]))

    def _open_outputs(self) -> None:
        config.OUTPUTS_DIR.mkdir(exist_ok=True)
        if sys.platform == "win32":
            os.startfile(config.OUTPUTS_DIR)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(config.OUTPUTS_DIR)])


def torch_cuda_ok() -> bool:
    import torch

    return torch.cuda.is_available()


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SemiReal Studio")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()
