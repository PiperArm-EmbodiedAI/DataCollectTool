from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tool_piper.collection.episode_manager import delete_episode, format_size, list_episodes
from tool_piper.constants import DEFAULT_ASSETS_ROOT, DEFAULT_LEROBOT_ROOT, DEFAULT_OUTPUTS_ROOT
from tool_piper.gui.formatting import (
    command_build_observation,
    command_convert,
    command_export_openpi_legacy,
    command_lerobot_check,
    command_norm_stats,
    command_policy_dry_run,
    command_raw_check,
    command_replay,
    pretty_json,
)
from tool_piper.gui.guide import copy_paths_guide, pc_training_guide, stage_hint
from tool_piper.gui.workers import TaskWorker, run_in_thread
from tool_piper.lerobot.convert import ConversionProgress, convert_raw_to_lerobot
from tool_piper.lerobot.inspect import check_lerobot_dataset
from tool_piper.lerobot.openpi_legacy import export_openpi_legacy_dataset
from tool_piper.lerobot.replay import make_replay_video
from tool_piper.model.observation import load_sample_observation, observation_summary
from tool_piper.norm.stats import compute_norm_stats
from tool_piper.raw.inspector import check_raw_dataset


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Piper Tool")
        self.resize(1180, 820)
        self.threads = []
        self.task_started_at = 0.0
        self.active_worker: TaskWorker | None = None
        self.active_task_name: str | None = None
        self.active_task_cleanup_paths: list[Path] = []
        self.task_finished_outcome = "idle"
        self._build_ui()
        self.refresh_episodes()
        self.update_guides()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_collect_tab(), "Collect")
        self.tabs.addTab(self._build_process_tab(), "Process")
        self.tabs.addTab(self._build_guide_tab(), "Guide / Export")
        root.addWidget(self.tabs, stretch=2)
        root.addWidget(self._build_status_panel())
        root.addWidget(self._build_shell_panel(), stretch=4)
        self.setCentralWidget(central)

        QShortcut(QKeySequence("S"), self, activated=self.start_collection_placeholder)
        QShortcut(QKeySequence("E"), self, activated=self.stop_collection_placeholder)

    def _build_collect_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        form_box = QGroupBox("Collection setup")
        form = QGridLayout(form_box)
        self.collect_raw_root = QLineEdit(str(Path("data/raw/pickup")))
        self.collect_task = QLineEdit("pick up the egg and put into the yellow plate")
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self.browse_directory(self.collect_raw_root))
        form.addWidget(QLabel("Raw root"), 0, 0)
        form.addWidget(self.collect_raw_root, 0, 1)
        form.addWidget(browse, 0, 2)
        form.addWidget(QLabel("Task"), 1, 0)
        form.addWidget(self.collect_task, 1, 1, 1, 2)
        layout.addWidget(form_box)

        controls = QHBoxLayout()
        start = QPushButton("Start Collection (S)")
        stop = QPushButton("Stop Collection (E)")
        refresh = QPushButton("Refresh Episodes")
        delete = QPushButton("Delete Selected Episode")
        open_folder = QPushButton("Open Episode Folder")
        start.clicked.connect(self.start_collection_placeholder)
        stop.clicked.connect(self.stop_collection_placeholder)
        refresh.clicked.connect(self.refresh_episodes)
        delete.clicked.connect(self.delete_selected_episode)
        open_folder.clicked.connect(self.open_selected_episode)
        for button in (start, stop, refresh, delete, open_folder):
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)

        preview = QHBoxLayout()
        self.camera_preview = QLabel("Camera preview: hardware backend not connected. Existing raw data can be managed here.")
        self.camera_preview.setAlignment(Qt.AlignCenter)
        self.camera_preview.setMinimumHeight(120)
        self.camera_preview.setStyleSheet("border: 1px solid #777; background: #222; color: #ddd;")
        self.robot_preview = QLabel("Robot state preview: reserved for CAN/Piper SDK integration.")
        self.robot_preview.setAlignment(Qt.AlignCenter)
        self.robot_preview.setMinimumHeight(120)
        self.robot_preview.setStyleSheet("border: 1px solid #777; background: #222; color: #ddd;")
        preview.addWidget(self.camera_preview)
        preview.addWidget(self.robot_preview)
        layout.addLayout(preview)

        self.episode_table = QTableWidget(0, 5)
        self.episode_table.setHorizontalHeaderLabels(["Episode", "Modified", "Size", "Files", "Path"])
        self.episode_table.horizontalHeader().setStretchLastSection(True)
        self.episode_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.episode_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.episode_table)
        return page

    def _build_process_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        paths = QGroupBox("Paths and options")
        form = QFormLayout(paths)
        self.raw_root = QLineEdit(str(Path("data/raw/pickup")))
        self.repo_id = QLineEdit("piper_tool_pickup_100")
        self.task = QLineEdit("pick up the egg and put into the yellow plate")
        self.lerobot_root = QLineEdit(str(DEFAULT_LEROBOT_ROOT / self.repo_id.text()))
        self.assets_root = QLineEdit(str(DEFAULT_ASSETS_ROOT))
        self.outputs_root = QLineEdit(str(DEFAULT_OUTPUTS_ROOT))
        self.latest_n = QSpinBox()
        self.latest_n.setRange(0, 100000)
        self.latest_n.setValue(100)
        self.latest_n.setSpecialValueText("all")
        self.episode_index = QSpinBox()
        self.episode_index.setRange(0, 100000)
        self.frame_index = QSpinBox()
        self.frame_index.setRange(0, 100000000)
        self.max_frames = QSpinBox()
        self.max_frames.setRange(0, 100000000)
        self.max_frames.setSpecialValueText("all")
        self.policy_host = QLineEdit("127.0.0.1")
        self.policy_port = QSpinBox()
        self.policy_port.setRange(1, 65535)
        self.policy_port.setValue(8000)
        self.api_key = QLineEdit()
        self.api_key.setPlaceholderText("optional")

        self.repo_id.textChanged.connect(self._repo_id_changed)
        self.task.textChanged.connect(self.update_guides)
        self.raw_root.textChanged.connect(lambda text: self.collect_raw_root.setText(text))

        form.addRow("Raw root", self._with_browse(self.raw_root))
        form.addRow("Repo ID", self.repo_id)
        form.addRow("Task prompt", self.task)
        form.addRow("LeRobot root", self._with_browse(self.lerobot_root))
        form.addRow("Assets root", self._with_browse(self.assets_root))
        form.addRow("Outputs root", self._with_browse(self.outputs_root))
        form.addRow("Latest N", self.latest_n)
        form.addRow("Episode index", self.episode_index)
        form.addRow("Frame index", self.frame_index)
        form.addRow("Max frames", self.max_frames)
        form.addRow("Policy host", self.policy_host)
        form.addRow("Policy port", self.policy_port)
        form.addRow("API key", self.api_key)
        layout.addWidget(paths)

        buttons = QGridLayout()
        actions = [
            ("1. Check Raw Data", self.run_raw_check),
            ("2. Convert to LeRobot", self.run_convert),
            ("3. Check LeRobot Dataset", self.run_lerobot_check),
            ("4. Export OpenPI Legacy", self.run_export_openpi_legacy),
            ("5. Generate Replay Video", self.run_replay),
            ("6. Compute Norm Stats", self.run_norm_stats),
            ("7. Build Observation", self.run_build_observation),
            ("Optional: Policy Dry Run", self.run_policy_dry_run),
            ("Open Output Folder", lambda: self.open_path(Path(self.outputs_root.text()))),
            ("Clear Current Outputs", self.clear_current_outputs),
        ]
        for i, (label, slot) in enumerate(actions):
            button = QPushButton(label)
            button.clicked.connect(slot)
            buttons.addWidget(button, i // 2, i % 2)
        layout.addLayout(buttons)
        layout.addStretch()
        return page

    def _build_guide_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.guide_text = QTextEdit()
        self.guide_text.setReadOnly(True)
        copy = QPushButton("Copy Guide")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.guide_text.toPlainText()))
        refresh = QPushButton("Refresh Guide")
        refresh.clicked.connect(self.update_guides)
        row = QHBoxLayout()
        row.addWidget(refresh)
        row.addWidget(copy)
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(self.guide_text)
        return page

    def _build_status_panel(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("0/0 | elapsed 00:00:00 | ETA --:--:--")
        self.cancel_task_button = QPushButton("Cancel Task")
        self.cancel_task_button.setEnabled(False)
        self.cancel_task_button.clicked.connect(self.cancel_active_task)
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(90)
        layout.addWidget(self.status_label, 0, 0)
        layout.addWidget(self.progress_bar, 0, 1)
        layout.addWidget(self.progress_label, 0, 2)
        layout.addWidget(self.cancel_task_button, 0, 3)
        layout.addWidget(self.result_text, 1, 0, 1, 4)
        return panel

    def _build_shell_panel(self) -> QWidget:
        box = QGroupBox("Shell / command feedback")
        layout = QVBoxLayout(box)
        toolbar = QHBoxLayout()
        clear = QPushButton("Clear")
        copy = QPushButton("Copy All")
        save = QPushButton("Save Log")
        self.autoscroll = QCheckBox("Auto scroll")
        self.autoscroll.setChecked(True)
        clear.clicked.connect(lambda: self.shell.clear())
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.shell.toPlainText()))
        save.clicked.connect(self.save_shell_log)
        for widget in (clear, copy, save, self.autoscroll):
            toolbar.addWidget(widget)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.shell = QPlainTextEdit()
        self.shell.setReadOnly(True)
        self.shell.setMinimumHeight(260)
        self.shell.setMaximumBlockCount(10000)
        self.shell.setStyleSheet("background: #111; color: #d7ffd7; font-family: monospace;")
        layout.addWidget(self.shell)
        return box

    def _with_browse(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self.browse_directory(line_edit))
        layout.addWidget(button)
        return widget

    def _repo_id_changed(self, text: str) -> None:
        self.lerobot_root.setText(str(DEFAULT_LEROBOT_ROOT / text))
        self.update_guides()

    def config_values(self) -> dict[str, Any]:
        latest = self.latest_n.value() or None
        max_frames = self.max_frames.value() or None
        return {
            "raw_root": Path(self.raw_root.text()),
            "repo_id": self.repo_id.text().strip(),
            "task": self.task.text(),
            "dataset_root": Path(self.lerobot_root.text()),
            "assets_root": Path(self.assets_root.text()),
            "outputs_root": Path(self.outputs_root.text()),
            "latest_n": latest,
            "episode_index": self.episode_index.value(),
            "frame_index": self.frame_index.value(),
            "max_frames": max_frames,
            "policy_host": self.policy_host.text().strip(),
            "policy_port": self.policy_port.value(),
            "api_key": self.api_key.text().strip() or None,
        }

    def browse_directory(self, line_edit: QLineEdit) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select directory", line_edit.text() or str(Path.cwd()))
        if directory:
            line_edit.setText(directory)

    def append_shell(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.shell.appendPlainText(f"[{timestamp}] {message}")
        if self.autoscroll.isChecked():
            self.shell.verticalScrollBar().setValue(self.shell.verticalScrollBar().maximum())

    def save_shell_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save log", "piper_tool_gui.log", "Log files (*.log);;Text files (*.txt)")
        if path:
            Path(path).write_text(self.shell.toPlainText())

    def open_path(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            QMessageBox.warning(self, "Missing path", f"Path does not exist:\n{path}")
            return
        subprocess.Popen(["xdg-open", str(path)])

    def clear_current_outputs(self) -> None:
        cfg = self.config_values()
        repo_id = cfg["repo_id"]
        paths = [
            cfg["dataset_root"],
            cfg["dataset_root"].parent / f"{repo_id}_openpi_legacy",
            cfg["assets_root"] / repo_id,
            cfg["assets_root"] / f"{repo_id}_openpi_legacy",
            cfg["outputs_root"] / "replay" / repo_id,
        ]
        existing_paths = [path for path in paths if path.exists()]
        self.append_shell("$ clear generated outputs for current repo")
        self.append_shell("raw root is preserved: " + str(cfg["raw_root"]))
        if not existing_paths:
            self.append_shell("No generated outputs found for current repo.")
            QMessageBox.information(self, "Nothing to clear", "No generated outputs found for the current repo.")
            return
        message = "Delete these generated outputs? Raw data will not be touched.\n\n" + "\n".join(
            str(path) for path in existing_paths
        )
        reply = QMessageBox.question(self, "Clear generated outputs", message)
        if reply != QMessageBox.Yes:
            self.append_shell("Clear generated outputs cancelled.")
            return
        deleted = self._delete_generated_paths(existing_paths, "manual clear")
        self.status_label.setText("Cleared generated outputs for current repo")
        self.result_text.setPlainText("Deleted generated outputs:\n" + "\n".join(str(path) for path in deleted))
        self.update_guides()

    def _delete_generated_paths(self, paths: list[Path], reason: str) -> list[Path]:
        deleted = []
        self.append_shell(f"Cleaning generated outputs ({reason}). Raw data is preserved.")
        for path in paths:
            path = Path(path)
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
                parent = path.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            deleted.append(path)
            self.append_shell(f"Deleted generated output: {path}")
        if not deleted:
            self.append_shell("No generated cleanup targets existed.")
        return deleted

    def start_task(self, name: str, equivalent_command: str, backend: str, fn, on_result=None, cleanup_paths=None) -> None:
        if self.active_worker is not None:
            QMessageBox.information(self, "Task running", "A task is already running. Cancel it or wait for it to finish first.")
            return
        self.task_started_at = time.monotonic()
        self.active_task_name = name
        self.active_task_cleanup_paths = list(cleanup_paths or [])
        self.task_finished_outcome = "running"
        self.status_label.setText(f"Running: {name}")
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("running | elapsed 00:00:00 | ETA --:--:--")
        self.result_text.clear()
        self.append_shell(f"$ {equivalent_command}")
        self.append_shell(f"calling {backend}")
        worker_holder: dict[str, TaskWorker] = {}

        def task_wrapper():
            return fn(worker_holder["worker"])

        worker = TaskWorker(task_wrapper)
        worker_holder["worker"] = worker
        self.active_worker = worker
        self.cancel_task_button.setEnabled(True)
        worker.signals.message.connect(self.append_shell)
        worker.signals.progress.connect(self._task_progress)
        worker.signals.result.connect(lambda result: self._task_result(name, result, on_result))
        worker.signals.error.connect(lambda error: self._task_error(name, error))
        worker.signals.cancelled.connect(lambda reason: self._task_cancelled(name, reason))
        worker.signals.finished.connect(lambda: self._task_finished(name))
        thread = run_in_thread(worker)
        self.threads.append(thread)

    def _task_progress(self, current: int, total: int, label: str, eta: str) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(current)
        elapsed = time.monotonic() - self.task_started_at
        self.progress_label.setText(f"{current}/{total} | elapsed {self.format_seconds(elapsed)} | ETA {eta}")
        self.append_shell(f"{label} | elapsed {self.format_seconds(elapsed)} | ETA {eta}")

    def _task_result(self, name: str, result: object, on_result) -> None:
        self.task_finished_outcome = "success"
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.result_text.setPlainText(pretty_json(result))
        self.append_shell(f"{name} result:\n{pretty_json(result)}")
        if on_result is not None:
            on_result(result)

    def _task_error(self, name: str, error: str) -> None:
        self.task_finished_outcome = "error"
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Failed: {name}")
        self.result_text.setPlainText(error)
        self.append_shell(f"{name} failed:\n{error}")

    def _task_cancelled(self, name: str, reason: str) -> None:
        self.task_finished_outcome = "cancelled"
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Cancelled: {name}")
        self.append_shell(f"{name} cancelled: {reason}")
        deleted = self._delete_generated_paths(self.active_task_cleanup_paths, f"cancelled {name}")
        lines = [f"Cancelled: {name}", "Raw data preserved."]
        if deleted:
            lines.append("Deleted generated outputs:")
            lines.extend(str(path) for path in deleted)
        else:
            lines.append("No generated outputs needed cleanup.")
        self.result_text.setPlainText("\n".join(lines))

    def _task_finished(self, name: str) -> None:
        elapsed = time.monotonic() - self.task_started_at
        if self.task_finished_outcome == "success":
            self.status_label.setText(f"Finished: {name} in {self.format_seconds(elapsed)}")
            self.append_shell(f"{name} finished in {self.format_seconds(elapsed)}")
        elif self.task_finished_outcome == "cancelled":
            self.append_shell(f"{name} stopped in {self.format_seconds(elapsed)}")
        elif self.task_finished_outcome == "error":
            self.append_shell(f"{name} failed in {self.format_seconds(elapsed)}")
        self.cancel_task_button.setEnabled(False)
        self.active_worker = None
        self.active_task_name = None
        self.active_task_cleanup_paths = []
        self.task_finished_outcome = "idle"

    def cancel_active_task(self) -> None:
        if self.active_worker is None or self.active_task_name is None:
            return
        self.status_label.setText(f"Cancelling: {self.active_task_name}")
        self.append_shell(f"Cancellation requested for {self.active_task_name}")
        self.cancel_task_button.setEnabled(False)
        self.active_worker.cancel()

    def format_seconds(self, seconds: float | None) -> str:
        if seconds is None:
            return "--:--:--"
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def run_raw_check(self) -> None:
        cfg = self.config_values()
        command = command_raw_check(str(cfg["raw_root"]))
        def task(worker: TaskWorker):
            def progress(current: int, total: int, label: str, eta_seconds: float | None) -> None:
                worker.signals.progress.emit(current, total, label, self.format_seconds(eta_seconds))

            return check_raw_dataset(cfg["raw_root"], progress=progress, cancel_check=worker.cancel_token.raise_if_cancelled)

        self.start_task(
            "Raw Check",
            command,
            "tool_piper.raw.inspector.check_raw_dataset(...) ",
            task,
            lambda _: self.show_stage_hint("raw-check"),
        )

    def run_convert(self) -> None:
        cfg = self.config_values()
        command = command_convert(str(cfg["raw_root"]), cfg["repo_id"], cfg["task"], cfg["latest_n"])

        def task(worker: TaskWorker):
            def progress(event: ConversionProgress) -> None:
                label = f"[{event.current}/{event.total}] {event.message}"
                eta = self.format_seconds(event.eta_seconds)
                worker.signals.progress.emit(event.current, event.total, label, eta)

            return convert_raw_to_lerobot(
                raw_root=cfg["raw_root"],
                repo_id=cfg["repo_id"],
                task=cfg["task"],
                output_root=cfg["dataset_root"],
                latest_n=cfg["latest_n"],
                progress=progress,
                cancel_check=worker.cancel_token.raise_if_cancelled,
            )

        self.start_task(
            "Convert to LeRobot",
            command,
            "tool_piper.lerobot.convert.convert_raw_to_lerobot(...) ",
            task,
            lambda _: self.show_stage_hint("convert"),
            cleanup_paths=[cfg["dataset_root"]],
        )

    def run_lerobot_check(self) -> None:
        cfg = self.config_values()
        command = command_lerobot_check(cfg["repo_id"], str(cfg["dataset_root"]))
        def task(worker: TaskWorker):
            def progress(current: int, total: int, label: str, eta_seconds: float | None) -> None:
                worker.signals.progress.emit(current, total, label, self.format_seconds(eta_seconds))

            return check_lerobot_dataset(
                cfg["repo_id"], cfg["dataset_root"], progress=progress, cancel_check=worker.cancel_token.raise_if_cancelled
            )

        self.start_task(
            "LeRobot Check",
            command,
            "tool_piper.lerobot.inspect.check_lerobot_dataset(...) ",
            task,
            lambda _: self.show_stage_hint("lerobot-check"),
        )

    def run_export_openpi_legacy(self) -> None:
        cfg = self.config_values()
        output_root = cfg["dataset_root"].parent / f"{cfg['repo_id']}_openpi_legacy"
        command = command_export_openpi_legacy(cfg["repo_id"], str(cfg["dataset_root"]), cfg["task"], str(output_root))

        def task(worker: TaskWorker):
            def progress(current: int, total: int, label: str, eta_seconds: float | None) -> None:
                worker.signals.progress.emit(current, total, label, self.format_seconds(eta_seconds))

            return export_openpi_legacy_dataset(
                cfg["dataset_root"],
                cfg["repo_id"],
                cfg["task"],
                output_root,
                progress=progress,
                cancel_check=worker.cancel_token.raise_if_cancelled,
            )

        self.start_task(
            "Export OpenPI Legacy",
            command,
            "tool_piper.lerobot.openpi_legacy.export_openpi_legacy_dataset(...) ",
            task,
            lambda _: self.show_stage_hint("openpi-legacy"),
            cleanup_paths=[output_root],
        )

    def run_replay(self) -> None:
        cfg = self.config_values()
        command = command_replay(cfg["repo_id"], str(cfg["dataset_root"]), cfg["episode_index"], cfg["max_frames"])
        out_path = cfg["outputs_root"] / "replay" / cfg["repo_id"] / f"episode_{cfg['episode_index']:03d}.mp4"
        def task(worker: TaskWorker):
            def progress(current: int, total: int, label: str, eta_seconds: float | None) -> None:
                worker.signals.progress.emit(current, total, label, self.format_seconds(eta_seconds))

            return make_replay_video(
                cfg["repo_id"],
                cfg["dataset_root"],
                cfg["episode_index"],
                out_path,
                cfg["max_frames"],
                progress=progress,
                cancel_check=worker.cancel_token.raise_if_cancelled,
            )

        self.start_task(
            "Generate Replay",
            command,
            "tool_piper.lerobot.replay.make_replay_video(...) ",
            task,
            lambda _: self.show_stage_hint("replay"),
            cleanup_paths=[out_path],
        )

    def run_norm_stats(self) -> None:
        cfg = self.config_values()
        out_dir = cfg["assets_root"] / cfg["repo_id"]
        command = command_norm_stats(cfg["repo_id"], str(cfg["dataset_root"]), str(out_dir))
        out_path = out_dir / "norm_stats.json"

        def task(worker: TaskWorker):
            def progress(current: int, total: int, label: str, eta_seconds: float | None) -> None:
                worker.signals.progress.emit(current, total, label, self.format_seconds(eta_seconds))

            return compute_norm_stats(
                cfg["repo_id"],
                cfg["dataset_root"],
                out_dir,
                progress=progress,
                cancel_check=worker.cancel_token.raise_if_cancelled,
            )

        self.start_task(
            "Compute Norm Stats",
            command,
            "tool_piper.norm.stats.compute_norm_stats(...) ",
            task,
            lambda _: self.show_stage_hint("norm-stats"),
            cleanup_paths=[out_path],
        )

    def run_build_observation(self) -> None:
        cfg = self.config_values()
        command = command_build_observation(cfg["repo_id"], str(cfg["dataset_root"]), cfg["frame_index"], cfg["task"])

        def task(worker: TaskWorker):
            worker.cancel_token.raise_if_cancelled()
            observation = load_sample_observation(cfg["repo_id"], cfg["dataset_root"], cfg["frame_index"], cfg["task"])
            worker.cancel_token.raise_if_cancelled()
            return observation_summary(observation)

        self.start_task(
            "Build Observation",
            command,
            "tool_piper.model.observation.load_sample_observation(...) ",
            task,
            lambda _: self.show_stage_hint("observation"),
        )

    def run_policy_dry_run(self) -> None:
        cfg = self.config_values()
        command = command_policy_dry_run(
            cfg["policy_host"], cfg["policy_port"], cfg["repo_id"], str(cfg["dataset_root"]), cfg["frame_index"], cfg["task"]
        )

        def task(worker: TaskWorker):
            from tool_piper.model.policy_client import policy_dry_run

            worker.cancel_token.raise_if_cancelled()
            observation = load_sample_observation(cfg["repo_id"], cfg["dataset_root"], cfg["frame_index"], cfg["task"])
            worker.cancel_token.raise_if_cancelled()
            actions = policy_dry_run(cfg["policy_host"], cfg["policy_port"], observation, cfg["api_key"])
            worker.cancel_token.raise_if_cancelled()
            return {"actions_shape": tuple(actions.shape), "actions_preview": actions[:1].tolist()}

        self.start_task(
            "Policy Dry Run",
            command,
            "tool_piper.model.policy_client.policy_dry_run(...) ",
            task,
            lambda _: self.show_stage_hint("policy"),
        )

    def show_stage_hint(self, stage: str) -> None:
        cfg = self.config_values()
        hint = stage_hint(stage, cfg["repo_id"], cfg["task"], cfg["dataset_root"], cfg["assets_root"], cfg["outputs_root"])
        self.append_shell(hint)
        self.update_guides()

    def update_guides(self) -> None:
        if not hasattr(self, "guide_text"):
            return
        cfg = self.config_values() if hasattr(self, "raw_root") else {}
        if not cfg:
            return
        text = "\n".join(
            [
                pc_training_guide(cfg["repo_id"], cfg["task"]),
                copy_paths_guide(cfg["repo_id"], cfg["dataset_root"], cfg["assets_root"]),
            ]
        )
        self.guide_text.setPlainText(text)

    def refresh_episodes(self) -> None:
        raw_root = Path(self.collect_raw_root.text())
        episodes = list_episodes(raw_root)
        self.episode_table.setRowCount(len(episodes))
        for row, episode in enumerate(episodes):
            values = [
                episode.name,
                episode.modified_at.strftime("%Y-%m-%d %H:%M:%S"),
                format_size(episode.size_bytes),
                str(episode.file_count),
                str(episode.path),
            ]
            for col, value in enumerate(values):
                self.episode_table.setItem(row, col, QTableWidgetItem(value))
        self.append_shell(f"Episode scan: {len(episodes)} episodes under {raw_root}")

    def selected_episode_path(self) -> Path | None:
        rows = self.episode_table.selectionModel().selectedRows()
        if not rows:
            return None
        path_item = self.episode_table.item(rows[0].row(), 4)
        if path_item is None:
            return None
        return Path(path_item.text())

    def delete_selected_episode(self) -> None:
        path = self.selected_episode_path()
        if path is None:
            QMessageBox.information(self, "No selection", "Select an episode first.")
            return
        reply = QMessageBox.question(self, "Delete episode", f"Delete this episode?\n{path}")
        if reply != QMessageBox.Yes:
            return
        delete_episode(path)
        self.append_shell(f"Deleted episode: {path}")
        self.refresh_episodes()

    def open_selected_episode(self) -> None:
        path = self.selected_episode_path()
        if path is None:
            QMessageBox.information(self, "No selection", "Select an episode first.")
            return
        self.open_path(path)

    def start_collection_placeholder(self) -> None:
        self.append_shell("Start Collection requested. Hardware backend is not connected in this GUI version.")
        self.status_label.setText("Collection start requested: hardware backend not connected")

    def stop_collection_placeholder(self) -> None:
        self.append_shell("Stop Collection requested. Hardware backend is not connected in this GUI version.")
        self.status_label.setText("Collection stop requested: hardware backend not connected")
