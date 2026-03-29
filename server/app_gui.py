#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SyncML Studio — Interface graphique Qt pour l'alignement inter-flux deep learning.
Enveloppe le moteur IA.py avec un UI/UX complet.
"""

from __future__ import annotations

import sys
import json
import traceback
import time
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

# Ajoute la racine du projet au chemin pour trouver utils/ et pipeline/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Qt ────────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QTextEdit, QScrollArea, QFrame, QSizePolicy, QMessageBox,
    QHeaderView, QSlider, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QStatusBar, QToolBar, QLineEdit,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize,
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QIcon, QTextCursor, QPixmap,
)

# ── Matplotlib ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── Moteur IA ─────────────────────────────────────────────────────────────────
# Import différé dans les workers pour ne pas bloquer le démarrage de l'UI


# ══════════════════════════════════════════════════════════════════════════════
# Palette sombre
# ══════════════════════════════════════════════════════════════════════════════

DARK = {
    "bg":          "#0e0e14",
    "bg2":         "#16161f",
    "bg3":         "#1c1c2a",
    "border":      "#2a2a3d",
    "text":        "#dde1f0",
    "text_dim":    "#7a7d99",
    "accent":      "#5c7cfa",
    "accent2":     "#7c3aed",
    "green":       "#22c55e",
    "orange":      "#f59e0b",
    "red":         "#ef4444",
    "yellow":      "#eab308",
}

MPL_STYLE = {
    "figure.facecolor":    DARK["bg2"],
    "axes.facecolor":      DARK["bg3"],
    "axes.edgecolor":      DARK["border"],
    "axes.labelcolor":     DARK["text"],
    "axes.grid":           True,
    "grid.color":          DARK["border"],
    "grid.alpha":          0.5,
    "text.color":          DARK["text"],
    "xtick.color":         DARK["text_dim"],
    "ytick.color":         DARK["text_dim"],
    "lines.linewidth":     1.8,
    "font.size":           9,
}
plt.rcParams.update(MPL_STYLE)

CONF_CMAP = LinearSegmentedColormap.from_list(
    "conf", [(0, DARK["red"]), (0.62, DARK["orange"]), (1.0, DARK["green"])]
)


def conf_color(c: float) -> str:
    if c >= 0.75:
        return DARK["green"]
    if c >= 0.62:
        return DARK["orange"]
    return DARK["red"]


def apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    bg   = QColor(DARK["bg"])
    bg2  = QColor(DARK["bg2"])
    bg3  = QColor(DARK["bg3"])
    txt  = QColor(DARK["text"])
    dim  = QColor(DARK["text_dim"])
    acc  = QColor(DARK["accent"])
    brd  = QColor(DARK["border"])

    pal.setColor(QPalette.ColorRole.Window,          bg)
    pal.setColor(QPalette.ColorRole.WindowText,      txt)
    pal.setColor(QPalette.ColorRole.Base,            bg2)
    pal.setColor(QPalette.ColorRole.AlternateBase,   bg3)
    pal.setColor(QPalette.ColorRole.ToolTipBase,     bg3)
    pal.setColor(QPalette.ColorRole.ToolTipText,     txt)
    pal.setColor(QPalette.ColorRole.Text,            txt)
    pal.setColor(QPalette.ColorRole.Button,          bg3)
    pal.setColor(QPalette.ColorRole.ButtonText,      txt)
    pal.setColor(QPalette.ColorRole.BrightText,      QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link,            acc)
    pal.setColor(QPalette.ColorRole.Highlight,       acc)
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, dim)
    app.setPalette(pal)

    app.setStyleSheet(f"""
    QMainWindow, QDialog {{ background: {DARK["bg"]}; }}
    QWidget {{ background: {DARK["bg"]}; color: {DARK["text"]}; }}
    QGroupBox {{
        border: 1px solid {DARK["border"]};
        border-radius: 6px;
        margin-top: 10px;
        padding: 8px 6px 6px 6px;
        font-weight: 600;
        color: {DARK["text_dim"]};
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}

    QPushButton {{
        background: {DARK["bg3"]};
        border: 1px solid {DARK["border"]};
        border-radius: 5px;
        padding: 5px 14px;
        color: {DARK["text"]};
        font-weight: 500;
    }}
    QPushButton:hover  {{ background: {DARK["accent"]}; border-color: {DARK["accent"]}; color: #fff; }}
    QPushButton:pressed {{ background: {DARK["accent2"]}; }}
    QPushButton:disabled {{ color: {DARK["text_dim"]}; background: {DARK["bg2"]}; }}

    QPushButton#primary {{
        background: {DARK["accent"]};
        border-color: {DARK["accent"]};
        color: #fff;
        font-weight: 700;
        padding: 7px 18px;
    }}
    QPushButton#primary:hover {{ background: {DARK["accent2"]}; }}
    QPushButton#danger {{
        background: {DARK["red"]};
        border-color: {DARK["red"]};
        color: #fff;
    }}

    QTabWidget::pane {{
        border: 1px solid {DARK["border"]};
        border-radius: 4px;
        background: {DARK["bg2"]};
    }}
    QTabBar::tab {{
        background: {DARK["bg3"]};
        border: 1px solid {DARK["border"]};
        border-bottom: none;
        border-radius: 4px 4px 0 0;
        padding: 6px 16px;
        color: {DARK["text_dim"]};
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{ background: {DARK["bg2"]}; color: {DARK["text"]}; border-bottom: 2px solid {DARK["accent"]}; }}
    QTabBar::tab:hover    {{ color: {DARK["text"]}; }}

    QTableWidget {{
        background: {DARK["bg2"]};
        alternate-background-color: {DARK["bg3"]};
        gridline-color: {DARK["border"]};
        border: 1px solid {DARK["border"]};
        border-radius: 4px;
        selection-background-color: {DARK["accent"]};
    }}
    QHeaderView::section {{
        background: {DARK["bg3"]};
        border: none;
        border-right: 1px solid {DARK["border"]};
        border-bottom: 1px solid {DARK["border"]};
        padding: 5px 8px;
        font-weight: 600;
        color: {DARK["text_dim"]};
    }}

    QListWidget {{
        background: {DARK["bg2"]};
        border: 1px solid {DARK["border"]};
        border-radius: 4px;
        alternate-background-color: {DARK["bg3"]};
    }}
    QListWidget::item:selected {{ background: {DARK["accent"]}; color: #fff; }}
    QListWidget::item:hover    {{ background: {DARK["bg3"]}; }}

    QTextEdit, QLineEdit {{
        background: {DARK["bg2"]};
        border: 1px solid {DARK["border"]};
        border-radius: 4px;
        color: {DARK["text"]};
        selection-background-color: {DARK["accent"]};
    }}
    QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {DARK["bg2"]};
        border: 1px solid {DARK["border"]};
        border-radius: 4px;
        padding: 3px 6px;
        color: {DARK["text"]};
    }}
    QProgressBar {{
        background: {DARK["bg3"]};
        border: 1px solid {DARK["border"]};
        border-radius: 4px;
        text-align: center;
        color: {DARK["text"]};
    }}
    QProgressBar::chunk {{ background: {DARK["accent"]}; border-radius: 3px; }}

    QScrollBar:vertical {{
        background: {DARK["bg2"]};
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{ background: {DARK["border"]}; border-radius: 4px; min-height: 20px; }}
    QScrollBar::handle:vertical:hover {{ background: {DARK["accent"]}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

    QSplitter::handle {{ background: {DARK["border"]}; width: 1px; height: 1px; }}

    QStatusBar {{ background: {DARK["bg3"]}; border-top: 1px solid {DARK["border"]}; color: {DARK["text_dim"]}; }}

    QCheckBox::indicator {{
        width: 14px; height: 14px;
        background: {DARK["bg2"]};
        border: 1px solid {DARK["border"]};
        border-radius: 3px;
    }}
    QCheckBox::indicator:checked {{ background: {DARK["accent"]}; border-color: {DARK["accent"]}; }}
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Widgets utilitaires
# ══════════════════════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvas):
    """Canvas matplotlib réutilisable avec fond sombre."""

    def __init__(self, figsize=(8, 4), dpi=96):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor=DARK["bg2"])
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background: {DARK['bg2']}; border: none;")

    def clear(self):
        self.fig.clear()
        self.draw()


class LogWidget(QTextEdit):
    """Journal console avec couleurs."""

    LEVELS = {
        "INFO":    DARK["text"],
        "OK":      DARK["green"],
        "WARN":    DARK["orange"],
        "ERROR":   DARK["red"],
        "TRAIN":   DARK["accent"],
        "DIM":     DARK["text_dim"],
    }

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Menlo, Consolas, monospace", 10))
        self.setStyleSheet(f"background:{DARK['bg2']}; border:1px solid {DARK['border']}; border-radius:4px;")

    def log(self, msg: str, level: str = "INFO"):
        color = self.LEVELS.get(level, DARK["text"])
        ts = time.strftime("%H:%M:%S")
        html = (
            f'<span style="color:{DARK["text_dim"]}">[{ts}]</span> '
            f'<span style="color:{color}">{msg}</span>'
        )
        self.append(html)
        self.moveCursor(QTextCursor.MoveOperation.End)


class ConfidenceBadge(QLabel):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.setValue(value)
        self.setFixedWidth(56)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("Arial", 9, QFont.Weight.Bold))

    def setValue(self, v: float):
        col = conf_color(v)
        self.setText(f"{v:.3f}")
        self.setStyleSheet(
            f"background:{col}22; color:{col}; "
            f"border:1px solid {col}55; border-radius:4px; padding:2px 4px;"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Workers QThread
# ══════════════════════════════════════════════════════════════════════════════

class DiscoveryWorker(QThread):
    """Découvre les sessions dans /Volumes/T9/data/."""
    result   = pyqtSignal(list)   # list of session Path strings
    error    = pyqtSignal(str)

    def run(self):
        try:
            from pipeline.pipeline import INGEST_DIR
            sessions = sorted(
                [s for s in INGEST_DIR.iterdir()
                 if s.is_dir() and not s.name.startswith("_")],
                key=lambda p: p.name,
            )
            self.result.emit([str(s) for s in sessions])
        except Exception:
            self.error.emit(traceback.format_exc())


class TrainWorker(QThread):
    epoch_done   = pyqtSignal(int, float)     # epoch, loss
    log_msg      = pyqtSignal(str, str)       # msg, level
    pseudo_stats = pyqtSignal(int, int, int)  # total, pos, neg
    finished_ok  = pyqtSignal(str)            # model_dir path
    error        = pyqtSignal(str)

    def __init__(self, sessions: list, params: dict):
        super().__init__()
        self.sessions = sessions  # list of Path strings
        self.params   = params

    def run(self):
        try:
            import utils.sync as ia
            import torch
            from torch.utils.data import DataLoader
            from pipeline.pipeline import MODEL_DIR

            ia.set_seed()
            sess = [Path(s) for s in self.sessions]

            self.log_msg.emit(f"Chargement de {len(sess)} session(s)…", "INFO")
            examples = ia.build_training_examples(
                sessions   = sess,
                resample_ms= self.params["resample_ms"],
                max_lag_ms = self.params["max_lag_ms"],
                window_ms  = self.params["window_ms"],
            )

            if len(examples) < 50:
                raise RuntimeError(f"Seulement {len(examples)} exemples pseudo-labelisés. Données insuffisantes.")

            pos = sum(y for _, _, y, _ in examples)
            neg = len(examples) - pos
            self.pseudo_stats.emit(len(examples), pos, neg)
            self.log_msg.emit(f"Pseudo-labels: total={len(examples)}  pos={pos}  neg={neg}", "OK")

            ds = ia.PairWindowDataset(examples)
            dl = DataLoader(ds, batch_size=self.params["batch_size"], shuffle=True, drop_last=False)

            model = ia.CrossModalAligner().to(ia.DEVICE)
            opt   = torch.optim.AdamW(
                model.parameters(),
                lr           = self.params["lr"],
                weight_decay = ia.WEIGHT_DECAY,
            )

            import torch.nn.functional as F

            epochs = self.params["epochs"]
            for epoch in range(epochs):
                model.train()
                losses = []
                for xa, xb, y in dl:
                    xa = xa.to(ia.DEVICE)
                    xb = xb.to(ia.DEVICE)
                    y  = y.to(ia.DEVICE)
                    opt.zero_grad()
                    logit, ea, eb = model(xa, xb)
                    bce  = F.binary_cross_entropy_with_logits(logit, y)
                    ctr  = ia.contrastive_margin(ea, eb, y)
                    loss = 0.75 * bce + 0.25 * ctr
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))

                mean_loss = float(np.mean(losses))
                self.epoch_done.emit(epoch + 1, mean_loss)
                self.log_msg.emit(
                    f"Époque {epoch+1:02d}/{epochs}  loss={mean_loss:.4f}", "TRAIN"
                )

            model_dir = MODEL_DIR
            ia.save_model(model, model_dir)
            self.log_msg.emit(f"Modèle sauvegardé → {model_dir}", "OK")
            self.finished_ok.emit(str(model_dir))

        except Exception:
            self.error.emit(traceback.format_exc())


class InferenceWorker(QThread):
    pair_done  = pyqtSignal(dict)  # serialized PairEstimate
    log_msg    = pyqtSignal(str, str)
    finished_ok= pyqtSignal(list)  # list of dicts
    error      = pyqtSignal(str)

    def __init__(self, session: str, params: dict, apply: bool = False, dry_run: bool = True):
        super().__init__()
        self.session = session
        self.params  = params
        self.apply   = apply
        self.dry_run = dry_run

    def run(self):
        try:
            import utils.sync as ia
            from pipeline.pipeline import MODEL_DIR
            sess_p    = Path(self.session)
            model_dir = MODEL_DIR

            if not (model_dir / "model.pt").exists():
                raise RuntimeError("Aucun modèle trouvé. Lancez d'abord l'entraînement.")

            model  = ia.load_model(model_dir)
            fluxes = ia.load_all_fluxes(sess_p)

            estimates = []
            for ref_name, tgt_name in ia.PAIRS:
                if ref_name not in fluxes or tgt_name not in fluxes:
                    self.log_msg.emit(f"Paire {ref_name}↔{tgt_name} ignorée (flux manquant)", "WARN")
                    continue

                est = ia.estimate_pair_offset(
                    model      = model,
                    ref        = fluxes[ref_name],
                    tgt        = fluxes[tgt_name],
                    resample_ms= self.params["resample_ms"],
                    max_lag_ms = self.params["max_lag_ms"],
                    window_ms  = self.params["window_ms"],
                )

                d = {
                    "ref_name":          est.ref_name,
                    "tgt_name":          est.tgt_name,
                    "delta_start_ms":    est.delta_start_ms,
                    "residual_ms":       est.residual_ms,
                    "total_offset_ms":   est.total_offset_ms,
                    "shift_to_apply_ms": est.shift_to_apply_ms,
                    "confidence":        est.confidence,
                    "peak_margin":       est.peak_margin,
                    "best_score":        est.best_score,
                    "second_score":      est.second_score,
                    "is_reliable":       est.is_reliable,
                    "method":            est.method,
                    "lags_ms":           est.lags_ms.tolist(),
                    "scores":            est.scores.tolist(),
                }
                estimates.append(d)
                self.pair_done.emit(d)
                self.log_msg.emit(
                    f"{ref_name} ↔ {tgt_name}  shift={est.shift_to_apply_ms:+.1f} ms  "
                    f"conf={est.confidence:.3f}  reliable={est.is_reliable}",
                    "OK" if est.is_reliable else "WARN",
                )

            if self.apply and not self.dry_run:
                applied = set()
                for est_d in estimates:
                    if not est_d["is_reliable"]:
                        continue
                    if est_d["tgt_name"] in applied:
                        continue
                    ia.apply_shift_to_target(sess_p, est_d["tgt_name"], est_d["shift_to_apply_ms"], dry_run=False)
                    applied.add(est_d["tgt_name"])
                    self.log_msg.emit(
                        f"Offset appliqué: {est_d['tgt_name']}  {est_d['shift_to_apply_ms']:+.1f} ms", "OK"
                    )

            self.finished_ok.emit(estimates)

        except Exception:
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# Panel : Ingestion
# ══════════════════════════════════════════════════════════════════════════════

class IngestionPanel(QWidget):
    sessions_changed = pyqtSignal(list)  # session list (paths in /Volumes/T9/data)

    def __init__(self, log: LogWidget):
        super().__init__()
        self.log       = log
        self._sessions = []
        self._worker   = None
        self._build_ui()

    def _build_ui(self):
        from pipeline.pipeline import INGEST_DIR, SILVER_DIR
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Chemins hardcodés ──
        grp_paths = QGroupBox("Chemins de la pipeline")
        fl = QFormLayout(grp_paths)
        for label, path in [
            ("Travail (ingest)", str(INGEST_DIR)),
            ("Sortie (silver)",  str(SILVER_DIR)),
        ]:
            lbl = QLabel(path)
            lbl.setStyleSheet(f"color:{DARK['accent']}; font-family:monospace;")
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            fl.addRow(label + ":", lbl)
        layout.addWidget(grp_paths)

        # ── Bouton scan ──
        self.btn_scan = QPushButton("Scanner les sessions")
        self.btn_scan.setObjectName("primary")
        self.btn_scan.clicked.connect(self._scan)
        layout.addWidget(self.btn_scan)

        # ── Liste des sessions ──
        grp_sess = QGroupBox("Sessions dans /Volumes/T9/data/")
        vl = QVBoxLayout(grp_sess)
        self.lbl_count = QLabel("— sessions")
        self.lbl_count.setStyleSheet(f"color:{DARK['text_dim']};")
        vl.addWidget(self.lbl_count)

        self.list_sessions = QListWidget()
        self.list_sessions.setAlternatingRowColors(True)
        self.list_sessions.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_sessions.setMinimumHeight(120)
        vl.addWidget(self.list_sessions)

        btn_sel_all = QPushButton("Tout sélectionner")
        btn_sel_all.clicked.connect(self.list_sessions.selectAll)
        vl.addWidget(btn_sel_all)
        layout.addWidget(grp_sess)

        # ── Métadonnées session sélectionnée ──
        grp_meta = QGroupBox("Métadonnées session")
        vl2 = QVBoxLayout(grp_meta)
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setMaximumHeight(160)
        self.meta_text.setFont(QFont("Menlo, Consolas, monospace", 9))
        vl2.addWidget(self.meta_text)
        layout.addWidget(grp_meta)
        layout.addStretch()

        self.list_sessions.currentItemChanged.connect(self._show_meta)

    def _scan(self):
        self.btn_scan.setEnabled(False)
        self.lbl_count.setText("Scan en cours…")
        self._worker = DiscoveryWorker()
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, sessions: list):
        self._sessions = sessions
        self.list_sessions.clear()
        for s in sessions:
            name = Path(s).name
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, s)
            has_tracker = (Path(s) / "tracker_positions.csv").exists()
            has_meta    = (Path(s) / "metadata.json").exists()
            # Statut pipeline (pipeline_state.json dans le même dossier)
            ps_path = Path(s) / "pipeline_state.json"
            pipeline_done = False
            if ps_path.exists():
                try:
                    import json as _json
                    ps = _json.loads(ps_path.read_text())
                    pipeline_done = ps.get("finished", False) and ps.get("success", False)
                except Exception:
                    pass
            if pipeline_done:
                item.setForeground(QColor(DARK["green"]))
                item.setText(name + "  ✓")
            elif has_tracker and has_meta:
                item.setForeground(QColor(DARK["text"]))
            else:
                item.setForeground(QColor(DARK["text_dim"]))
                item.setText(name + "  ⚠")
            self.list_sessions.addItem(item)

        self.lbl_count.setText(f"{len(sessions)} session(s) trouvée(s)")
        self.log.log(f"[Ingestion] {len(sessions)} sessions dans /Volumes/T9/data/", "OK")
        self.btn_scan.setEnabled(True)
        self.list_sessions.selectAll()
        self.sessions_changed.emit(sessions)

    def _on_error(self, err: str):
        self.log.log(f"[Ingestion] ERREUR: {err}", "ERROR")
        self.btn_scan.setEnabled(True)
        self.lbl_count.setText("Erreur")

    def _show_meta(self, item: QListWidgetItem):
        if item is None:
            return
        sess_path = Path(item.data(Qt.ItemDataRole.UserRole))
        meta_file = sess_path / "metadata.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text())
                self.meta_text.setText(json.dumps(data, indent=2, ensure_ascii=False))
                return
            except Exception:
                pass
        files = sorted(f.name for f in sess_path.rglob("*") if f.is_file())
        self.meta_text.setText("\n".join(files[:40]))

    def get_selected_sessions(self) -> list:
        selected = self.list_sessions.selectedItems()
        if not selected:
            return self._sessions
        return [item.data(Qt.ItemDataRole.UserRole) for item in selected]


# ══════════════════════════════════════════════════════════════════════════════
# Panel : Entraînement
# ══════════════════════════════════════════════════════════════════════════════

class TrainPanel(QWidget):
    model_ready = pyqtSignal()

    def __init__(self, log: LogWidget, ingestion: IngestionPanel):
        super().__init__()
        self.log        = log
        self.ingestion  = ingestion
        self._worker    = None
        self._losses    = []
        self._epochs    = []
        self._t_start   = 0.0
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ── Colonne gauche : paramètres + contrôles ──
        left = QVBoxLayout()
        left.setSpacing(8)

        grp_hp = QGroupBox("Hyperparamètres")
        form = QFormLayout(grp_hp)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.sp_epochs     = QSpinBox();     self.sp_epochs.setRange(1, 200);   self.sp_epochs.setValue(18)
        self.sp_batch      = QSpinBox();     self.sp_batch.setRange(8, 512);    self.sp_batch.setValue(64)
        self.sp_lr         = QDoubleSpinBox(); self.sp_lr.setDecimals(6); self.sp_lr.setRange(1e-6, 1e-1); self.sp_lr.setValue(1e-3); self.sp_lr.setSingleStep(1e-4)
        self.sp_resample   = QDoubleSpinBox(); self.sp_resample.setDecimals(1); self.sp_resample.setRange(1.0, 50.0); self.sp_resample.setValue(5.0)
        self.sp_max_lag    = QDoubleSpinBox(); self.sp_max_lag.setDecimals(0); self.sp_max_lag.setRange(50.0, 2000.0); self.sp_max_lag.setValue(400.0)
        self.sp_window     = QDoubleSpinBox(); self.sp_window.setDecimals(0); self.sp_window.setRange(500.0, 10000.0); self.sp_window.setValue(2200.0)

        form.addRow("Époques",       self.sp_epochs)
        form.addRow("Batch size",    self.sp_batch)
        form.addRow("Learning rate", self.sp_lr)
        form.addRow("Resample (ms)", self.sp_resample)
        form.addRow("Max lag (ms)",  self.sp_max_lag)
        form.addRow("Fenêtre (ms)",  self.sp_window)
        left.addWidget(grp_hp)

        # Pseudo-stats
        grp_ps = QGroupBox("Pseudo-labels")
        fl = QFormLayout(grp_ps)
        self.lbl_total = QLabel("—");  fl.addRow("Total",  self.lbl_total)
        self.lbl_pos   = QLabel("—");  fl.addRow("Positif",self.lbl_pos)
        self.lbl_neg   = QLabel("—");  fl.addRow("Négatif",self.lbl_neg)
        left.addWidget(grp_ps)

        # Progression
        grp_prog = QGroupBox("Progression")
        vl = QVBoxLayout(grp_prog)
        self.lbl_epoch   = QLabel("Époque: —")
        self.lbl_loss    = QLabel("Loss: —")
        self.lbl_eta     = QLabel("ETA: —")
        self.prog_bar    = QProgressBar()
        self.prog_bar.setValue(0)
        for lbl in (self.lbl_epoch, self.lbl_loss, self.lbl_eta):
            lbl.setStyleSheet(f"color:{DARK['text_dim']};")
            vl.addWidget(lbl)
        vl.addWidget(self.prog_bar)
        left.addWidget(grp_prog)

        self.btn_train = QPushButton("Lancer l'entraînement")
        self.btn_train.setObjectName("primary")
        self.btn_train.clicked.connect(self._start_train)
        left.addWidget(self.btn_train)

        self.lbl_model_status = QLabel("Aucun modèle")
        self.lbl_model_status.setStyleSheet(f"color:{DARK['text_dim']}; font-style:italic;")
        left.addWidget(self.lbl_model_status)

        left.addStretch()
        left_w = QWidget(); left_w.setLayout(left); left_w.setFixedWidth(280)
        layout.addWidget(left_w)

        # ── Colonne droite : courbe de loss ──
        right = QVBoxLayout()
        lbl_plot = QLabel("Courbe de perte (loss)")
        lbl_plot.setStyleSheet(f"color:{DARK['text_dim']}; font-weight:600;")
        right.addWidget(lbl_plot)

        self.canvas_loss = MplCanvas(figsize=(6, 4))
        right.addWidget(self.canvas_loss)
        layout.addLayout(right)

    def _get_params(self) -> dict:
        return {
            "epochs":     self.sp_epochs.value(),
            "batch_size": self.sp_batch.value(),
            "lr":         self.sp_lr.value(),
            "resample_ms":self.sp_resample.value(),
            "max_lag_ms": self.sp_max_lag.value(),
            "window_ms":  self.sp_window.value(),
        }

    def _start_train(self):
        sessions = self.ingestion.get_selected_sessions()
        if not sessions:
            QMessageBox.warning(self, "Attention", "Sélectionnez au moins une session.")
            return

        self._losses.clear()
        self._epochs.clear()
        self._t_start = time.time()
        epochs = self.sp_epochs.value()
        self.prog_bar.setMaximum(epochs)
        self.prog_bar.setValue(0)
        self.btn_train.setEnabled(False)
        self.btn_train.setText("Entraînement…")

        self._worker = TrainWorker(sessions, self._get_params())
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.log_msg.connect(self.log.log)
        self._worker.pseudo_stats.connect(self._on_pseudo)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_pseudo(self, total, pos, neg):
        self.lbl_total.setText(str(total))
        self.lbl_pos.setText(str(pos))
        self.lbl_neg.setText(str(neg))

    def _on_epoch(self, epoch: int, loss: float):
        self._losses.append(loss)
        self._epochs.append(epoch)

        total_epochs = self.sp_epochs.value()
        self.prog_bar.setValue(epoch)
        self.lbl_epoch.setText(f"Époque: {epoch}/{total_epochs}")
        self.lbl_loss.setText(f"Loss: {loss:.4f}")

        elapsed = time.time() - self._t_start
        if epoch > 0:
            per_ep  = elapsed / epoch
            remain  = per_ep * (total_epochs - epoch)
            self.lbl_eta.setText(f"ETA: {remain:.0f}s")

        self._redraw_loss()

    def _redraw_loss(self):
        fig = self.canvas_loss.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK["bg3"])

        xs = self._epochs
        ys = self._losses
        ax.plot(xs, ys, color=DARK["accent"], lw=2, label="Train loss")

        # Smooth
        if len(ys) >= 5:
            from scipy.ndimage import gaussian_filter1d
            sm = gaussian_filter1d(ys, sigma=1.5)
            ax.plot(xs, sm, color=DARK["green"], lw=1.5, linestyle="--", alpha=0.8, label="Lissée")

        ax.set_xlabel("Époque")
        ax.set_ylabel("Loss")
        ax.set_title("Courbe d'entraînement", color=DARK["text"])
        ax.legend(facecolor=DARK["bg3"], edgecolor=DARK["border"], labelcolor=DARK["text"])
        if len(ys) > 1:
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        self.canvas_loss.draw()

    def _on_done(self, model_dir: str):
        self.btn_train.setEnabled(True)
        self.btn_train.setText("Lancer l'entraînement")
        self.prog_bar.setValue(self.sp_epochs.value())
        self.lbl_model_status.setText(f"✓ Modèle: {Path(model_dir).name}")
        self.lbl_model_status.setStyleSheet(f"color:{DARK['green']};")
        self.model_ready.emit()

    def _on_error(self, err: str):
        self.log.log(err, "ERROR")
        self.btn_train.setEnabled(True)
        self.btn_train.setText("Lancer l'entraînement")
        QMessageBox.critical(self, "Erreur d'entraînement", err[:400])

    def get_params(self) -> dict:
        return self._get_params()


# ══════════════════════════════════════════════════════════════════════════════
# Panel : Inférence
# ══════════════════════════════════════════════════════════════════════════════

class InferencePanel(QWidget):
    estimates_ready = pyqtSignal(list, str)  # estimates, session_path

    def __init__(self, log: LogWidget, ingestion: IngestionPanel, train_panel: TrainPanel):
        super().__init__()
        self.log         = log
        self.ingestion   = ingestion
        self.train_panel = train_panel
        self._worker     = None
        self._estimates  = []
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ── Colonne gauche ──
        left = QVBoxLayout()
        left.setSpacing(8)

        grp_sess = QGroupBox("Session cible")
        vl = QVBoxLayout(grp_sess)
        self.combo_session = QComboBox()
        self.combo_session.setPlaceholderText("Choisir une session…")
        vl.addWidget(self.combo_session)
        left.addWidget(grp_sess)

        grp_opts = QGroupBox("Options")
        fl = QFormLayout(grp_opts)
        self.chk_apply   = QCheckBox("Appliquer les offsets")
        self.chk_dry_run = QCheckBox("Dry-run (simulation)")
        self.chk_dry_run.setChecked(True)
        fl.addRow(self.chk_apply)
        fl.addRow(self.chk_dry_run)
        left.addWidget(grp_opts)

        self.btn_run = QPushButton("Estimer les offsets")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run)
        left.addWidget(self.btn_run)

        self.btn_apply_all = QPushButton("Appliquer les offsets fiables")
        self.btn_apply_all.setObjectName("danger")
        self.btn_apply_all.setEnabled(False)
        self.btn_apply_all.clicked.connect(self._apply_all)
        left.addWidget(self.btn_apply_all)

        grp_sum = QGroupBox("Résumé")
        fl2 = QFormLayout(grp_sum)
        self.lbl_pairs_ok = QLabel("—")
        self.lbl_pairs_ko = QLabel("—")
        self.lbl_mean_conf= QLabel("—")
        fl2.addRow("Fiables:", self.lbl_pairs_ok)
        fl2.addRow("Non fiables:", self.lbl_pairs_ko)
        fl2.addRow("Conf. moy.:", self.lbl_mean_conf)
        left.addWidget(grp_sum)
        left.addStretch()

        left_w = QWidget(); left_w.setLayout(left); left_w.setFixedWidth(260)
        layout.addWidget(left_w)

        # ── Colonne droite : table des paires ──
        right = QVBoxLayout()
        lbl = QLabel("Résultats par paire")
        lbl.setStyleSheet(f"color:{DARK['text_dim']}; font-weight:600;")
        right.addWidget(lbl)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Référence", "Cible", "Δstart (ms)", "Résidu (ms)",
            "Shift (ms)", "Confiance", "Margin", "Fiable"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        right.addWidget(self.table)
        layout.addLayout(right)

    def refresh_sessions(self, sessions: list):
        self.combo_session.clear()
        for s in sessions:
            self.combo_session.addItem(Path(s).name, s)

    def _run(self):
        idx = self.combo_session.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "Attention", "Sélectionnez une session.")
            return

        sess = self.combo_session.itemData(idx)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Estimation…")
        self.table.setRowCount(0)
        self._estimates = []

        apply   = self.chk_apply.isChecked()
        dry_run = self.chk_dry_run.isChecked()

        params = self.train_panel.get_params()
        self._worker = InferenceWorker(sess, params, apply=apply, dry_run=dry_run)
        self._worker.pair_done.connect(self._on_pair)
        self._worker.log_msg.connect(self.log.log)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_pair(self, d: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)

        cells = [
            d["ref_name"], d["tgt_name"],
            f"{d['delta_start_ms']:+.1f}",
            f"{d['residual_ms']:+.1f}",
            f"{d['shift_to_apply_ms']:+.1f}",
        ]
        for col, txt in enumerate(cells):
            item = QTableWidgetItem(txt)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)

        # Confiance avec couleur
        conf   = d["confidence"]
        c_item = QTableWidgetItem(f"{conf:.3f}")
        c_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        c_item.setForeground(QColor(conf_color(conf)))
        self.table.setItem(row, 5, c_item)

        # Margin
        m_item = QTableWidgetItem(f"{d['peak_margin']:.3f}")
        m_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 6, m_item)

        # Fiable
        rel_txt = "✓" if d["is_reliable"] else "✗"
        r_item  = QTableWidgetItem(rel_txt)
        r_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        r_item.setForeground(QColor(DARK["green"] if d["is_reliable"] else DARK["red"]))
        self.table.setItem(row, 7, r_item)

        self._estimates.append(d)

    def _on_done(self, estimates: list):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Estimer les offsets")
        if estimates:
            ok   = sum(1 for e in estimates if e["is_reliable"])
            ko   = len(estimates) - ok
            mean = np.mean([e["confidence"] for e in estimates])
            self.lbl_pairs_ok.setText(f"{ok}")
            self.lbl_pairs_ko.setText(f"{ko}")
            self.lbl_mean_conf.setText(f"{mean:.3f}")
            self.lbl_pairs_ok.setStyleSheet(f"color:{DARK['green']};")
            self.lbl_pairs_ko.setStyleSheet(f"color:{DARK['red']};")
            sess_idx = self.combo_session.currentIndex()
            sess_path = self.combo_session.itemData(sess_idx) if sess_idx >= 0 else ""
            self.estimates_ready.emit(estimates, sess_path)
            self.btn_apply_all.setEnabled(ok > 0)

    def _on_error(self, err: str):
        self.log.log(err, "ERROR")
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Estimer les offsets")
        QMessageBox.critical(self, "Erreur d'inférence", err[:400])

    def _apply_all(self):
        ok = sum(1 for e in self._estimates if e["is_reliable"])
        ans = QMessageBox.question(
            self, "Confirmation",
            f"Appliquer {ok} offset(s) fiable(s) sur les fichiers ?\nCette action est irréversible (sauvegarde .bak créée).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        sess_idx = self.combo_session.currentIndex()
        if sess_idx < 0:
            return
        sess = self.combo_session.itemData(sess_idx)
        params = self.train_panel.get_params()
        self._worker = InferenceWorker(sess, params, apply=True, dry_run=False)
        self._worker.pair_done.connect(lambda _: None)
        self._worker.log_msg.connect(self.log.log)
        self._worker.finished_ok.connect(lambda _: self.log.log("Offsets appliqués avec succès.", "OK"))
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self.btn_apply_all.setEnabled(False)


# ══════════════════════════════════════════════════════════════════════════════
# Panel : Visualisation
# ══════════════════════════════════════════════════════════════════════════════

class VizPanel(QWidget):
    def __init__(self, log: LogWidget):
        super().__init__()
        self.log = log
        self._estimates: list = []
        self._sess_path: str  = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # ── Tab 1 : Lag Score Curves ──
        self.canvas_lags = MplCanvas(figsize=(10, 5))
        tabs.addTab(self.canvas_lags, "Courbes de score (lags)")

        # ── Tab 2 : Confidence heatmap ──
        self.canvas_heatmap = MplCanvas(figsize=(8, 4))
        tabs.addTab(self.canvas_heatmap, "Heatmap confiance")

        # ── Tab 3 : Alignement signal ──
        self.canvas_align = MplCanvas(figsize=(10, 6))
        tabs.addTab(self.canvas_align, "Alignement des signaux")

        # ── Tab 4 : Vue d'ensemble session ──
        self.canvas_overview = MplCanvas(figsize=(10, 5))
        tabs.addTab(self.canvas_overview, "Vue d'ensemble")

        self._tabs = tabs
        self._tabs.currentChanged.connect(self._on_tab)

    def update_estimates(self, estimates: list, sess_path: str):
        self._estimates  = estimates
        self._sess_path  = sess_path
        self._draw_lags()
        self._draw_heatmap()
        self._draw_overview()

    def _on_tab(self, idx: int):
        if idx == 2 and self._estimates:
            self._draw_alignment()

    def _draw_lags(self):
        estimates = self._estimates
        if not estimates:
            return

        n = len(estimates)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig = self.canvas_lags.fig
        fig.clear()
        fig.patch.set_facecolor(DARK["bg2"])

        axes = fig.subplots(nrows, ncols, squeeze=False)

        for i, est in enumerate(estimates):
            ax = axes[i // ncols][i % ncols]
            ax.set_facecolor(DARK["bg3"])

            lags   = np.array(est["lags_ms"])
            scores = np.array(est["scores"])

            if len(lags) == 0:
                ax.text(0.5, 0.5, "Pas de données", ha="center", va="center",
                        color=DARK["text_dim"], transform=ax.transAxes)
            else:
                ax.plot(lags, scores, color=DARK["accent"], lw=1.8)
                ax.fill_between(lags, scores, alpha=0.15, color=DARK["accent"])
                best_lag = est["residual_ms"]
                ax.axvline(best_lag, color=DARK["red"], lw=2, linestyle="--",
                           label=f"best={best_lag:+.0f}ms")
                ax.axhline(0.62, color=DARK["orange"], lw=1, linestyle=":", alpha=0.7)

            col = conf_color(est["confidence"])
            ax.set_title(
                f"{est['ref_name']} ↔ {est['tgt_name']}\n"
                f"conf={est['confidence']:.3f}  reliable={'✓' if est['is_reliable'] else '✗'}",
                color=col, fontsize=8, pad=4
            )
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Lag (ms)", fontsize=7)
            ax.set_ylabel("Score", fontsize=7)
            ax.legend(fontsize=7, facecolor=DARK["bg3"], edgecolor=DARK["border"],
                      labelcolor=DARK["text"])

        # Masquer axes vides
        for i in range(n, nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        fig.suptitle("Score modèle en fonction du lag — par paire",
                     color=DARK["text"], fontsize=10, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas_lags.draw()

    def _draw_heatmap(self):
        estimates = self._estimates
        if not estimates:
            return

        fig = self.canvas_heatmap.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK["bg3"])

        pairs  = [f"{e['ref_name']}\n↔\n{e['tgt_name']}" for e in estimates]
        confs  = [e["confidence"] for e in estimates]
        shifts = [e["shift_to_apply_ms"] for e in estimates]

        x = np.arange(len(pairs))
        bars = ax.bar(x, confs, color=[conf_color(c) for c in confs],
                      edgecolor=DARK["border"], width=0.6)

        ax.axhline(0.62, color=DARK["orange"], lw=1.5, linestyle="--", label="Seuil fiable (0.62)")
        ax.axhline(0.75, color=DARK["green"],  lw=1.0, linestyle=":",  label="Seuil excellent (0.75)")

        for bar, shift, conf in zip(bars, shifts, confs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{shift:+.0f}ms",
                    ha="center", va="bottom", fontsize=8,
                    color=DARK["text"])

        ax.set_xticks(x)
        ax.set_xticklabels(pairs, fontsize=7.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Confiance")
        ax.set_title("Confiance par paire — offsets estimés", color=DARK["text"])
        ax.legend(facecolor=DARK["bg3"], edgecolor=DARK["border"], labelcolor=DARK["text"])
        fig.tight_layout()
        self.canvas_heatmap.draw()

    def _draw_alignment(self):
        """Affiche l'overlay ref/tgt pour chaque paire avec l'offset appliqué."""
        if not self._estimates or not self._sess_path:
            return

        try:
            import utils.sync as ia
            sess_p = Path(self._sess_path)
            fluxes = ia.load_all_fluxes(sess_p)
        except Exception as e:
            self.log.log(f"[Viz] Erreur chargement flux: {e}", "ERROR")
            return

        n = len(self._estimates)
        if n == 0:
            return

        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols

        fig = self.canvas_align.fig
        fig.clear()
        fig.patch.set_facecolor(DARK["bg2"])

        axes = fig.subplots(nrows, ncols, squeeze=False)

        for i, est in enumerate(self._estimates):
            ax = axes[i // ncols][i % ncols]
            ax.set_facecolor(DARK["bg3"])

            ref_name = est["ref_name"]
            tgt_name = est["tgt_name"]

            if ref_name not in fluxes or tgt_name not in fluxes:
                ax.text(0.5, 0.5, "Flux indisponible", ha="center", va="center",
                        color=DARK["text_dim"], transform=ax.transAxes)
                ax.set_title(f"{ref_name} ↔ {tgt_name}", color=DARK["text_dim"], fontsize=8)
                continue

            ref = fluxes[ref_name]
            tgt = fluxes[tgt_name]

            t_max = min(float(ref.t_ms_rel[-1]), float(tgt.t_ms_rel[-1]), 8000.0)
            t_grid = np.arange(0, t_max, 5.0)

            ref_sig = np.interp(t_grid, ref.t_ms_rel, ref.signal, left=0.0, right=0.0)

            shift = est["shift_to_apply_ms"]
            tgt_t = tgt.t_ms_rel + est["delta_start_ms"] + est["residual_ms"]
            tgt_sig = np.interp(t_grid, tgt_t, tgt.signal, left=0.0, right=0.0)

            ax.plot(t_grid / 1000, ref_sig, color=DARK["accent"], lw=1.5,
                    label=f"Réf: {ref_name}", alpha=0.9)
            ax.plot(t_grid / 1000, tgt_sig, color=DARK["green"], lw=1.5,
                    label=f"Cib: {tgt_name} (aligné)", alpha=0.9, linestyle="--")

            col = conf_color(est["confidence"])
            ax.set_title(
                f"{ref_name} ↔ {tgt_name}  shift={shift:+.0f}ms  conf={est['confidence']:.3f}",
                color=col, fontsize=8
            )
            ax.set_xlabel("Temps (s)", fontsize=7)
            ax.legend(fontsize=7, facecolor=DARK["bg3"], edgecolor=DARK["border"],
                      labelcolor=DARK["text"])
            ax.tick_params(labelsize=7)

        for i in range(n, nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        fig.suptitle("Alignement des signaux après correction",
                     color=DARK["text"], fontsize=10, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas_align.draw()

    def _draw_overview(self):
        estimates = self._estimates
        if not estimates:
            return

        fig = self.canvas_overview.fig
        fig.clear()
        fig.patch.set_facecolor(DARK["bg2"])

        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

        # ── Gauche : scatter confiance vs shift ──
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(DARK["bg3"])

        confs  = [e["confidence"]        for e in estimates]
        shifts = [e["shift_to_apply_ms"] for e in estimates]
        colors = [conf_color(c)           for c in confs]
        labels = [f"{e['tgt_name']}"      for e in estimates]

        sc = ax1.scatter(shifts, confs, c=confs, cmap=CONF_CMAP,
                         vmin=0, vmax=1, s=100, edgecolors=DARK["border"], linewidths=0.5, zorder=5)
        for x, y, lbl in zip(shifts, confs, labels):
            ax1.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4),
                         fontsize=7, color=DARK["text_dim"])

        ax1.axhline(0.62, color=DARK["orange"], lw=1.2, linestyle="--", alpha=0.8)
        ax1.axvline(0, color=DARK["text_dim"], lw=0.8, linestyle=":", alpha=0.5)
        ax1.set_xlabel("Shift à appliquer (ms)")
        ax1.set_ylabel("Confiance")
        ax1.set_title("Confiance vs Shift", color=DARK["text"])
        cb = fig.colorbar(sc, ax=ax1)
        cb.ax.tick_params(labelsize=7, colors=DARK["text_dim"])
        cb.set_label("Confiance", color=DARK["text_dim"], fontsize=8)

        # ── Droite : peak margin ──
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(DARK["bg3"])

        pairs   = [f"{e['ref_name'][:8]}\n↔\n{e['tgt_name'][:8]}" for e in estimates]
        margins = [e["peak_margin"] for e in estimates]
        bar_colors = [conf_color(e["confidence"]) for e in estimates]

        bars = ax2.barh(pairs, margins, color=bar_colors,
                        edgecolor=DARK["border"], height=0.5)
        ax2.axvline(0.06, color=DARK["orange"], lw=1.2, linestyle="--",
                    alpha=0.8, label="Seuil margin (0.06)")
        ax2.set_xlabel("Peak margin")
        ax2.set_title("Margin du pic de score", color=DARK["text"])
        ax2.legend(fontsize=7, facecolor=DARK["bg3"], edgecolor=DARK["border"],
                   labelcolor=DARK["text"])
        ax2.tick_params(labelsize=7)

        fig.suptitle("Vue d'ensemble — session", color=DARK["text"], fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas_overview.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Panel : Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class PipelineWorker(QThread):
    log_msg    = pyqtSignal(str, str)
    step_done  = pyqtSignal(dict)   # steps dict
    finished   = pyqtSignal(bool, str)  # success, silver_path

    def __init__(self, source_path: str, params: dict,
                 write_mode: bool, delete_after_store: bool = False):
        super().__init__()
        self.source_path       = source_path
        self.params            = params
        self.write_mode        = write_mode
        self.delete_after_store = delete_after_store

    def run(self):
        try:
            from pipeline.pipeline import PipelineRunner
            runner = PipelineRunner(
                source_path        = self.source_path,
                params             = self.params,
                write_mode         = self.write_mode,
                delete_after_store = self.delete_after_store,
                log_callback       = lambda msg, level="INFO": self.log_msg.emit(msg, level),
                resume             = True,
            )
            state = runner.run()
            self.finished.emit(state.success, state.silver_path or "")
        except Exception:
            self.log_msg.emit(traceback.format_exc(), "ERROR")
            self.finished.emit(False, "")


class PipelinePanel(QWidget):
    def __init__(self, log: LogWidget, ingestion: "IngestionPanel"):
        super().__init__()
        self.log       = log
        self.ingestion = ingestion
        self._worker   = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Sélection session ──
        grp_sess = QGroupBox("Session à traiter")
        vl = QVBoxLayout(grp_sess)
        self.combo_session = QComboBox()
        self.combo_session.setPlaceholderText("Choisir une session dans /Volumes/T9/data…")
        vl.addWidget(self.combo_session)
        layout.addWidget(grp_sess)

        # ── Options pipeline ──
        grp_opts = QGroupBox("Options")
        fl = QFormLayout(grp_opts)
        self.chk_write_mode = QCheckBox("Mode écriture — copier vers/home/ia/silver")
        self.chk_write_mode.setToolTip(
            "Si coché, la session validée sera copiée dans/home/ia/silver après traitement."
        )
        self.chk_delete_after = QCheckBox("Supprimer de /Volumes/T9/data/ après store")
        self.chk_delete_after.setToolTip(
            "ATTENTION : supprime définitivement la session de /Volumes/T9/data/ après copie vers silver.\n"
            "Nécessite que le mode écriture soit activé."
        )
        self.chk_delete_after.setEnabled(False)
        self.chk_write_mode.toggled.connect(self.chk_delete_after.setEnabled)
        self.chk_force_flux = QCheckBox("Forcer recalcul flux CSV")
        self.sp_resample = QDoubleSpinBox()
        self.sp_resample.setDecimals(1); self.sp_resample.setRange(1.0, 50.0); self.sp_resample.setValue(5.0)
        self.sp_max_lag = QDoubleSpinBox()
        self.sp_max_lag.setDecimals(0); self.sp_max_lag.setRange(50.0, 2000.0); self.sp_max_lag.setValue(400.0)
        self.sp_window = QDoubleSpinBox()
        self.sp_window.setDecimals(0); self.sp_window.setRange(500.0, 10000.0); self.sp_window.setValue(2200.0)
        fl.addRow(self.chk_write_mode)
        fl.addRow(self.chk_delete_after)
        fl.addRow(self.chk_force_flux)
        fl.addRow("Resample (ms)", self.sp_resample)
        fl.addRow("Max lag (ms)",  self.sp_max_lag)
        fl.addRow("Fenêtre (ms)",  self.sp_window)
        layout.addWidget(grp_opts)

        # ── Progression étapes ──
        grp_steps = QGroupBox("Étapes pipeline")
        steps_layout = QVBoxLayout(grp_steps)
        self._step_labels: Dict[str, QLabel] = {}
        STEP_NAMES = ["detect", "rotate", "tracker", "video", "verify_labels",
                      "flux_csv", "ia_sync", "validate", "store"]
        STEP_DISPLAY = {
            "detect":       "1. Détection",
            "rotate":       "2. Rotation",
            "tracker":      "3. Trackers",
            "video":        "4. Vidéo",
            "verify_labels":"5. Vérif. labels",
            "flux_csv":     "6. Flux CSV",
            "ia_sync":      "7. Sync IA",
            "validate":     "8. Validation",
            "store":        "9. Stockage silver",
        }
        for name in STEP_NAMES:
            row = QHBoxLayout()
            lbl_name = QLabel(STEP_DISPLAY[name])
            lbl_name.setFixedWidth(160)
            lbl_status = QLabel("—")
            lbl_status.setStyleSheet(f"color:{DARK['text_dim']};")
            self._step_labels[name] = lbl_status
            row.addWidget(lbl_name)
            row.addWidget(lbl_status)
            row.addStretch()
            steps_layout.addLayout(row)
        layout.addWidget(grp_steps)

        self.prog_bar = QProgressBar()
        self.prog_bar.setValue(0)
        layout.addWidget(self.prog_bar)

        self.btn_run = QPushButton("Lancer la pipeline")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)

        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        layout.addWidget(self.lbl_result)
        layout.addStretch()

    def refresh_sessions(self, sessions: list):
        self.combo_session.clear()
        for s in sessions:
            self.combo_session.addItem(Path(s).name, s)

    def _run(self):
        idx = self.combo_session.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "Attention", "Sélectionnez une session.")
            return

        source_path = self.combo_session.itemData(idx)
        write_mode         = self.chk_write_mode.isChecked()
        delete_after_store = self.chk_delete_after.isChecked() and write_mode

        if delete_after_store:
            from PyQt6.QtWidgets import QMessageBox
            ans = QMessageBox.warning(
                self, "Confirmation suppression",
                f"La session sera supprimée de /Volumes/T9/data/ après copie vers/home/ia/silver.\n\n"
                f"Session : {Path(source_path).name}\n\nContinuer ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        params = {
            "resample_ms": self.sp_resample.value(),
            "max_lag_ms":  self.sp_max_lag.value(),
            "window_ms":   self.sp_window.value(),
            "force_flux":  self.chk_force_flux.isChecked(),
        }

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Pipeline en cours…")
        self.prog_bar.setValue(0)
        self.lbl_result.setText("")
        for lbl in self._step_labels.values():
            lbl.setText("—")
            lbl.setStyleSheet(f"color:{DARK['text_dim']};")

        self._worker = PipelineWorker(source_path, params, write_mode, delete_after_store)
        self._worker.log_msg.connect(self.log.log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, success: bool, silver_path: str):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Lancer la pipeline")
        self.prog_bar.setValue(100 if success else self.prog_bar.value())
        if success:
            msg = "Pipeline terminée avec succès."
            if silver_path:
                msg += f"\nSilver: {silver_path}"
            self.lbl_result.setText(msg)
            self.lbl_result.setStyleSheet(f"color:{DARK['green']};")
        else:
            self.lbl_result.setText("Pipeline échouée. Consultez le journal.")
            self.lbl_result.setStyleSheet(f"color:{DARK['red']};")


# ══════════════════════════════════════════════════════════════════════════════
# Fenêtre principale
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SyncML Studio — Alignement inter-flux")
        self.resize(1400, 860)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header ──
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {DARK['bg3']}, stop:1 {DARK['bg2']});"
            f"border-bottom: 1px solid {DARK['border']};"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)

        lbl_title = QLabel("SyncML Studio")
        lbl_title.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        lbl_title.setStyleSheet(f"color:{DARK['accent']}; background:transparent;")

        lbl_sub = QLabel("Alignement inter-flux par deep learning auto-supervisé")
        lbl_sub.setStyleSheet(f"color:{DARK['text_dim']}; background:transparent; font-size:11px;")

        self.lbl_device = QLabel("…")
        self.lbl_device.setStyleSheet(f"color:{DARK['text_dim']}; background:transparent; font-size:10px;")
        QTimer.singleShot(200, self._set_device_label)

        hl.addWidget(lbl_title)
        hl.addWidget(lbl_sub)
        hl.addStretch()
        hl.addWidget(self.lbl_device)
        root_layout.addWidget(header)

        # ── Corps principal (splitter horizontal) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # ─ Panneau gauche : onglets fonctionnels ─
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setMinimumWidth(700)

        self.log = LogWidget()

        self.panel_ingest   = IngestionPanel(self.log)
        self.panel_pipeline = PipelinePanel(self.log, self.panel_ingest)
        self.panel_train    = TrainPanel(self.log, self.panel_ingest)
        self.panel_infer    = InferencePanel(self.log, self.panel_ingest, self.panel_train)
        self.panel_viz      = VizPanel(self.log)

        tabs.addTab(self.panel_ingest,   "1 · Ingestion")
        tabs.addTab(self.panel_pipeline, "2 · Pipeline")
        tabs.addTab(self.panel_train,    "3 · Entraînement")
        tabs.addTab(self.panel_infer,    "4 · Inférence")
        tabs.addTab(self.panel_viz,      "5 · Visualisation")

        splitter.addWidget(tabs)

        # ─ Panneau droit : log ─
        right_w = QWidget()
        rlay    = QVBoxLayout(right_w)
        rlay.setContentsMargins(6, 6, 6, 6)
        lbl_log = QLabel("Journal")
        lbl_log.setStyleSheet(f"color:{DARK['text_dim']}; font-weight:600; font-size:11px;")
        rlay.addWidget(lbl_log)
        rlay.addWidget(self.log)

        btn_clear = QPushButton("Effacer")
        btn_clear.clicked.connect(self.log.clear)
        rlay.addWidget(btn_clear)

        splitter.addWidget(right_w)
        splitter.setSizes([1050, 350])

        root_layout.addWidget(splitter, 1)

        # ── Barre de statut ──
        sb = QStatusBar()
        sb.setStyleSheet(f"background:{DARK['bg3']}; color:{DARK['text_dim']}; border-top:1px solid {DARK['border']};")
        self.setStatusBar(sb)
        self.sb_lbl = QLabel("Prêt")
        sb.addWidget(self.sb_lbl)

    def _connect_signals(self):
        self.panel_ingest.sessions_changed.connect(self.panel_infer.refresh_sessions)
        self.panel_ingest.sessions_changed.connect(self.panel_pipeline.refresh_sessions)
        self.panel_infer.estimates_ready.connect(self._on_estimates_ready)
        self.panel_train.model_ready.connect(lambda: self.sb_lbl.setText("Modèle entraîné et prêt."))

    def _on_estimates_ready(self, estimates: list, sess_path: str):
        self.panel_viz.update_estimates(estimates, sess_path)
        # Passer auto à l'onglet visualisation
        tabs = self.centralWidget().findChild(QTabWidget)
        if tabs:
            tabs.setCurrentIndex(4)

    def _set_device_label(self):
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            col = DARK["green"] if dev == "cuda" else DARK["text_dim"]
            self.lbl_device.setStyleSheet(f"color:{col}; background:transparent; font-size:10px;")
            self.lbl_device.setText(f"Device: {dev.upper()}")
        except Exception:
            self.lbl_device.setText("PyTorch non disponible")

    def closeEvent(self, event):
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SyncML Studio")

    apply_dark_palette(app)

    win = MainWindow()
    win.show()

    win.log.log("SyncML Studio démarré.", "OK")
    win.log.log("1. Déposez vos sessions dans /Volumes/T9/data/, puis scannez dans « Ingestion ».", "DIM")
    win.log.log("2. Lancez la pipeline dans « Pipeline » (mode safe par défaut).", "DIM")
    win.log.log("   → Cochez « Mode écriture » pour produire le résultat dans/home/ia/silver.", "DIM")
    win.log.log("   → Cochez « Supprimer de ingest » pour nettoyer après store (opt-in).", "DIM")
    win.log.log("3. Entraînez le modèle dans « Entraînement ».", "DIM")
    win.log.log("4. Estimez les offsets dans « Inférence ».", "DIM")
    win.log.log("5. Visualisez les résultats dans « Visualisation ».", "DIM")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
