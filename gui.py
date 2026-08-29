#!/usr/bin/env python3
"""Modern Desktop Studio GUI for Bank Loan Risk Prediction System.

Built with PySide6 (Qt6).
Author: Yordanos Andargachew (Phone: +251 952 190 305)
"""

import os
import sys
from typing import Any, Dict
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predictor import LoanRiskPredictor


class LoanRiskStudioGUI(QMainWindow):
    """Modern Desktop Studio application for Credit Risk Underwriting."""

    # Presets for fast interactive demonstration
    PRESETS: Dict[str, Dict[str, Any]] = {
        "Prime Low-Risk Borrower": {
            "Age": 48,
            "Sex": "male",
            "Job": 2,
            "Housing": "own",
            "Saving accounts": "rich",
            "Checking account": "rich",
            "Credit amount": 1500,
            "Duration": 12,
            "Purpose": "radio/TV",
        },
        "Subprime High-Risk Borrower": {
            "Age": 22,
            "Sex": "female",
            "Job": 1,
            "Housing": "rent",
            "Saving accounts": "little",
            "Checking account": "little",
            "Credit amount": 7500,
            "Duration": 48,
            "Purpose": "education",
        },
        "Moderate Business Borrower": {
            "Age": 38,
            "Sex": "male",
            "Job": 3,
            "Housing": "own",
            "Saving accounts": "moderate",
            "Checking account": "moderate",
            "Credit amount": 4500,
            "Duration": 24,
            "Purpose": "business",
        },
    }

    def __init__(self) -> None:
        super().__init__()
        self.is_dark_mode = True
        self.predictor = LoanRiskPredictor()

        self.setWindowTitle("Bank Loan Risk Prediction Studio")
        self.resize(1180, 820)
        self.setMinimumSize(950, 700)

        self._init_ui()
        self._apply_theme()

        # Load first preset as default
        self._load_preset("Prime Low-Risk Borrower")
        # Run initial assessment
        self._run_assessment()

    def _init_ui(self) -> None:
        """Construct the UI component layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Bar
        header_layout = QHBoxLayout()
        header_title_layout = QVBoxLayout()

        title_label = QLabel("🏦 Bank Loan Risk Prediction Studio")
        title_label.setFont(QFont("Inter, Roboto, Segoe UI, sans-serif", 18, QFont.Bold))
        self.title_label = title_label

        subtitle_label = QLabel("Intelligent Credit Scoring & Machine Learning Underwriting Engine | Author: Yordanos Andargachew (+251 952 190 305)")
        subtitle_label.setFont(QFont("Inter, Roboto, Segoe UI, sans-serif", 10))
        self.subtitle_label = subtitle_label

        header_title_layout.addWidget(title_label)
        header_title_layout.addWidget(subtitle_label)
        header_layout.addLayout(header_title_layout)

        header_layout.addStretch()

        # Theme Toggle Button
        self.theme_btn = QPushButton("🌙 Switch to Light Mode")
        self.theme_btn.setFixedWidth(190)
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.clicked.connect(self._toggle_theme)
        header_layout.addWidget(self.theme_btn)

        main_layout.addLayout(header_layout)

        # Main Splitter: Left Panel (Inputs & Presets), Right Panel (Decision & Metrics Dashboard)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left Container: Scrollable Input Form
        left_card = QFrame()
        left_card.setObjectName("formCard")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(14)

        # Preset Selector
        preset_box = QHBoxLayout()
        preset_lbl = QLabel("Quick Presets:")
        preset_lbl.setFont(QFont("Inter, Roboto, sans-serif", 10, QFont.Bold))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_box.addWidget(preset_lbl)
        preset_box.addWidget(self.preset_combo)
        left_layout.addLayout(preset_box)

        # Form Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        left_layout.addWidget(line)

        # Form Fields Grid
        grid = QGridLayout()
        grid.setVerticalSpacing(12)
        grid.setHorizontalSpacing(14)

        # Age
        grid.addWidget(QLabel("Applicant Age (years):"), 0, 0)
        self.spin_age = QSpinBox()
        self.spin_age.setRange(18, 100)
        self.spin_age.setValue(35)
        grid.addWidget(self.spin_age, 0, 1)

        # Sex
        grid.addWidget(QLabel("Biological Sex:"), 1, 0)
        self.combo_sex = QComboBox()
        self.combo_sex.addItems(["male", "female"])
        grid.addWidget(self.combo_sex, 1, 1)

        # Job
        grid.addWidget(QLabel("Employment / Job Role:"), 2, 0)
        self.combo_job = QComboBox()
        self.combo_job.addItem("0 - Unskilled (Non-Resident)", 0)
        self.combo_job.addItem("1 - Unskilled (Resident)", 1)
        self.combo_job.addItem("2 - Skilled Worker", 2)
        self.combo_job.addItem("3 - Highly Skilled / Management", 3)
        self.combo_job.setCurrentIndex(2)
        grid.addWidget(self.combo_job, 2, 1)

        # Housing
        grid.addWidget(QLabel("Housing Tenancy:"), 3, 0)
        self.combo_housing = QComboBox()
        self.combo_housing.addItems(["own", "rent", "free"])
        grid.addWidget(self.combo_housing, 3, 1)

        # Saving Accounts
        grid.addWidget(QLabel("Saving Accounts Balance:"), 4, 0)
        self.combo_saving = QComboBox()
        self.combo_saving.addItems(["little", "moderate", "quite rich", "rich", "unknown"])
        grid.addWidget(self.combo_saving, 4, 1)

        # Checking Account
        grid.addWidget(QLabel("Checking Account Status:"), 5, 0)
        self.combo_checking = QComboBox()
        self.combo_checking.addItems(["little", "moderate", "rich", "unknown"])
        grid.addWidget(self.combo_checking, 5, 1)

        # Credit Amount
        grid.addWidget(QLabel("Requested Amount (DM):"), 6, 0)
        self.spin_amount = QSpinBox()
        self.spin_amount.setRange(100, 50000)
        self.spin_amount.setSingleStep(500)
        self.spin_amount.setValue(5000)
        grid.addWidget(self.spin_amount, 6, 1)

        # Duration
        grid.addWidget(QLabel("Loan Duration (Months):"), 7, 0)
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(4, 72)
        self.spin_duration.setValue(24)
        grid.addWidget(self.spin_duration, 7, 1)

        # Purpose
        grid.addWidget(QLabel("Loan Purpose:"), 8, 0)
        self.combo_purpose = QComboBox()
        self.combo_purpose.addItems([
            "car",
            "furniture/equipment",
            "radio/TV",
            "domestic appliances",
            "repairs",
            "education",
            "business",
            "vacation/others",
        ])
        grid.addWidget(self.combo_purpose, 8, 1)

        left_layout.addLayout(grid)

        # Model Selector Radio Buttons
        model_group_box = QGroupBox("🤖 Machine Learning Scoring Engine")
        model_box_layout = QHBoxLayout(model_group_box)
        self.radio_rf = QRadioButton("Random Forest (Ensemble)")
        self.radio_rf.setChecked(True)
        self.radio_lr = QRadioButton("Logistic Regression (Baseline)")
        self.radio_rf.toggled.connect(self._run_assessment)
        self.radio_lr.toggled.connect(self._run_assessment)

        model_box_layout.addWidget(self.radio_rf)
        model_box_layout.addWidget(self.radio_lr)
        left_layout.addWidget(model_group_box)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.assess_btn = QPushButton("⚡ Assess Loan Risk")
        self.assess_btn.setFont(QFont("Inter, Roboto, sans-serif", 11, QFont.Bold))
        self.assess_btn.setMinimumHeight(44)
        self.assess_btn.setCursor(Qt.PointingHandCursor)
        self.assess_btn.clicked.connect(self._run_assessment)

        self.reset_btn = QPushButton("🔄 Reset Form")
        self.reset_btn.setMinimumHeight(44)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self._reset_form)

        btn_layout.addWidget(self.assess_btn, 3)
        btn_layout.addWidget(self.reset_btn, 1)
        left_layout.addLayout(btn_layout)

        content_layout.addWidget(left_card, 1)

        # Right Container: Result Dashboard
        right_card = QFrame()
        right_card.setObjectName("resultCard")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(16)

        # Verdict Header Card
        self.verdict_badge = QLabel("ASSESSING APPLICANT...")
        self.verdict_badge.setFont(QFont("Inter, Roboto, sans-serif", 15, QFont.Bold))
        self.verdict_badge.setAlignment(Qt.AlignCenter)
        self.verdict_badge.setMinimumHeight(60)
        self.verdict_badge.setObjectName("verdictBadge")
        right_layout.addWidget(self.verdict_badge)

        # Metrics Summary Grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)

        # Confidence Card
        self.card_conf = QFrame()
        self.card_conf.setObjectName("metricSubCard")
        c_layout = QVBoxLayout(self.card_conf)
        c_lbl = QLabel("Model Confidence")
        c_lbl.setFont(QFont("Inter, Roboto, sans-serif", 9))
        self.val_conf = QLabel("--%")
        self.val_conf.setFont(QFont("Inter, Roboto, sans-serif", 16, QFont.Bold))
        c_layout.addWidget(c_lbl)
        c_layout.addWidget(self.val_conf)
        metrics_grid.addWidget(self.card_conf, 0, 0)

        # Risk Tier Card
        self.card_tier = QFrame()
        self.card_tier.setObjectName("metricSubCard")
        t_layout = QVBoxLayout(self.card_tier)
        t_lbl = QLabel("Risk Classification Tier")
        t_lbl.setFont(QFont("Inter, Roboto, sans-serif", 9))
        self.val_tier = QLabel("--")
        self.val_tier.setFont(QFont("Inter, Roboto, sans-serif", 14, QFont.Bold))
        t_layout.addWidget(t_lbl)
        t_layout.addWidget(self.val_tier)
        metrics_grid.addWidget(self.card_tier, 0, 1)

        right_layout.addLayout(metrics_grid)

        # Probability Breakdown Bars
        prob_box = QGroupBox("📊 Calibrated Repayment vs Default Probabilities")
        prob_layout = QVBoxLayout(prob_box)
        prob_layout.setSpacing(10)

        # Good Probability (Repayment)
        prob_good_header = QHBoxLayout()
        prob_good_header.addWidget(QLabel("Good Repayment Probability:"))
        self.lbl_prob_good = QLabel("0.0%")
        self.lbl_prob_good.setFont(QFont("Inter, Roboto, sans-serif", 10, QFont.Bold))
        prob_good_header.addWidget(self.lbl_prob_good, alignment=Qt.AlignRight)
        prob_layout.addLayout(prob_good_header)

        self.bar_good = QProgressBar()
        self.bar_good.setRange(0, 100)
        self.bar_good.setTextVisible(False)
        self.bar_good.setObjectName("barGood")
        prob_layout.addWidget(self.bar_good)

        # Bad Probability (Default Risk)
        prob_bad_header = QHBoxLayout()
        prob_bad_header.addWidget(QLabel("Default / Delinquency Probability:"))
        self.lbl_prob_bad = QLabel("0.0%")
        self.lbl_prob_bad.setFont(QFont("Inter, Roboto, sans-serif", 10, QFont.Bold))
        prob_bad_header.addWidget(self.lbl_prob_bad, alignment=Qt.AlignRight)
        prob_layout.addLayout(prob_bad_header)

        self.bar_bad = QProgressBar()
        self.bar_bad.setRange(0, 100)
        self.bar_bad.setTextVisible(False)
        self.bar_bad.setObjectName("barBad")
        prob_layout.addWidget(self.bar_bad)

        right_layout.addWidget(prob_box)

        # Underwriting Recommendation Box
        rec_box = QGroupBox("📋 Automated Underwriting Recommendation")
        rec_layout = QVBoxLayout(rec_box)
        self.lbl_recommendation = QLabel("Awaiting evaluation...")
        self.lbl_recommendation.setWordWrap(True)
        self.lbl_recommendation.setFont(QFont("Inter, Roboto, sans-serif", 10))
        rec_layout.addWidget(self.lbl_recommendation)
        right_layout.addWidget(rec_box)

        # Metadata Footer
        self.lbl_meta = QLabel("Engine: Random Forest | Evaluated: --")
        self.lbl_meta.setFont(QFont("Inter, Roboto, sans-serif", 9))
        self.lbl_meta.setObjectName("metaLabel")
        right_layout.addWidget(self.lbl_meta)

        right_layout.addStretch()
        content_layout.addWidget(right_card, 1)

        main_layout.addLayout(content_layout)

        # Connect live updates on change
        self.spin_age.valueChanged.connect(self._run_assessment)
        self.spin_amount.valueChanged.connect(self._run_assessment)
        self.spin_duration.valueChanged.connect(self._run_assessment)
        self.combo_sex.currentIndexChanged.connect(self._run_assessment)
        self.combo_job.currentIndexChanged.connect(self._run_assessment)
        self.combo_housing.currentIndexChanged.connect(self._run_assessment)
        self.combo_saving.currentIndexChanged.connect(self._run_assessment)
        self.combo_checking.currentIndexChanged.connect(self._run_assessment)
        self.combo_purpose.currentIndexChanged.connect(self._run_assessment)

    def _get_applicant_data(self) -> Dict[str, Any]:
        """Extract user input values into dictionary."""
        job_val = self.combo_job.currentData()
        if job_val is None:
            job_val = self.combo_job.currentIndex()

        return {
            "Age": int(self.spin_age.value()),
            "Sex": str(self.combo_sex.currentText()),
            "Job": int(job_val),
            "Housing": str(self.combo_housing.currentText()),
            "Saving accounts": str(self.combo_saving.currentText()),
            "Checking account": str(self.combo_checking.currentText()),
            "Credit amount": int(self.spin_amount.value()),
            "Duration": int(self.spin_duration.value()),
            "Purpose": str(self.combo_purpose.currentText()),
        }

    def _on_preset_changed(self, preset_name: str) -> None:
        """Load selected preset."""
        if preset_name in self.PRESETS:
            self._load_preset(preset_name)
            self._run_assessment()

    def _load_preset(self, preset_name: str) -> None:
        """Populate form controls from preset."""
        preset = self.PRESETS[preset_name]
        self.spin_age.setValue(preset["Age"])
        self.combo_sex.setCurrentText(preset["Sex"])
        job_idx = self.combo_job.findData(preset["Job"])
        if job_idx >= 0:
            self.combo_job.setCurrentIndex(job_idx)
        self.combo_housing.setCurrentText(preset["Housing"])
        self.combo_saving.setCurrentText(preset["Saving accounts"])
        self.combo_checking.setCurrentText(preset["Checking account"])
        self.spin_amount.setValue(preset["Credit amount"])
        self.spin_duration.setValue(preset["Duration"])
        self.combo_purpose.setCurrentText(preset["Purpose"])

    def _reset_form(self) -> None:
        """Reset form inputs to standard defaults."""
        self._load_preset("Moderate Business Borrower")
        self._run_assessment()

    def _run_assessment(self) -> None:
        """Execute risk prediction and update UI dashboard."""
        applicant_data = self._get_applicant_data()
        model_name = "random_forest" if self.radio_rf.isChecked() else "logistic_regression"

        try:
            res = self.predictor.predict(applicant_data, model_name=model_name)
        except Exception as e:
            self.verdict_badge.setText(f"Error: {e}")
            return

        risk_class = res["risk_class"]
        risk_tier = res["risk_tier"]
        confidence = res["confidence_score"]
        prob_good = res["good_percentage"]
        prob_bad = res["bad_percentage"]
        rec = res["recommendation"]
        model_desc = res["model_used"]
        eval_time = res["evaluated_at"]

        # Update Verdict Badge
        if risk_class == "Good Risk":
            self.verdict_badge.setText("🟢 APPROVED: LOW / ACCEPTABLE CREDIT RISK")
            badge_style = """
                background-color: rgba(16, 185, 129, 0.18);
                color: #10b981;
                border: 2px solid #10b981;
                border-radius: 8px;
            """
        else:
            self.verdict_badge.setText("🔴 CAUTION: BAD RISK / DEFAULT PROPENSITY")
            badge_style = """
                background-color: rgba(239, 68, 68, 0.18);
                color: #ef4444;
                border: 2px solid #ef4444;
                border-radius: 8px;
            """
        self.verdict_badge.setStyleSheet(badge_style)

        # Update Metrics
        self.val_conf.setText(f"{confidence:.1f}%")
        self.val_tier.setText(risk_tier)
        self.val_tier.setStyleSheet(f"color: {res['badge_color']}; font-weight: bold;")

        # Update Bars
        self.lbl_prob_good.setText(f"{prob_good:.1f}%")
        self.bar_good.setValue(int(round(prob_good)))

        self.lbl_prob_bad.setText(f"{prob_bad:.1f}%")
        self.bar_bad.setValue(int(round(prob_bad)))

        # Update Recommendation
        self.lbl_recommendation.setText(rec)
        self.lbl_meta.setText(f"Engine: {model_desc} | Evaluated at: {eval_time}")

    def _toggle_theme(self) -> None:
        """Switch between dark mode and light mode."""
        self.is_dark_mode = not self.is_dark_mode
        self.theme_btn.setText("☀️ Switch to Dark Mode" if not self.is_dark_mode else "🌙 Switch to Light Mode")
        self._apply_theme()
        self._run_assessment()

    def _apply_theme(self) -> None:
        """Apply CSS stylesheet based on current theme."""
        if self.is_dark_mode:
            # Dark Mode Palette (#12121c base)
            style = """
                QMainWindow {
                    background-color: #12121c;
                }
                QWidget {
                    color: #f3f4f6;
                    font-family: 'Inter', 'Roboto', 'Segoe UI', sans-serif;
                }
                #formCard, #resultCard {
                    background-color: #1a1a2e;
                    border: 1px solid #282846;
                    border-radius: 12px;
                }
                #metricSubCard {
                    background-color: #23233c;
                    border: 1px solid #323254;
                    border-radius: 8px;
                    padding: 10px;
                }
                QGroupBox {
                    border: 1px solid #323254;
                    border-radius: 8px;
                    margin-top: 12px;
                    padding-top: 14px;
                    font-weight: bold;
                    color: #94a3b8;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px;
                }
                QLabel {
                    color: #e2e8f0;
                }
                #metaLabel {
                    color: #64748b;
                }
                QSpinBox, QComboBox {
                    background-color: #23233c;
                    color: #ffffff;
                    border: 1px solid #3c3c60;
                    border-radius: 6px;
                    padding: 6px 10px;
                    font-size: 13px;
                }
                QSpinBox:focus, QComboBox:focus {
                    border: 1px solid #6366f1;
                }
                QComboBox QAbstractItemView {
                    background-color: #1e1e34;
                    color: #ffffff;
                    selection-background-color: #4f46e5;
                    border: 1px solid #3c3c60;
                }
                QPushButton {
                    background-color: #23233c;
                    color: #ffffff;
                    border: 1px solid #3c3c60;
                    border-radius: 8px;
                    padding: 8px 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2d2d4e;
                    border-color: #6366f1;
                }
                QPushButton:pressed {
                    background-color: #1e1e34;
                }
                #assess_btn, QPushButton[text^="⚡"] {
                    background-color: #4f46e5;
                    color: #ffffff;
                    border: none;
                }
                QPushButton[text^="⚡"]:hover {
                    background-color: #4338ca;
                }
                QProgressBar {
                    background-color: #23233c;
                    border: 1px solid #323254;
                    border-radius: 6px;
                    height: 18px;
                }
                #barGood::chunk {
                    background-color: #10b981;
                    border-radius: 5px;
                }
                #barBad::chunk {
                    background-color: #ef4444;
                    border-radius: 5px;
                }
                QRadioButton {
                    color: #cbd5e1;
                    font-weight: normal;
                }
                QRadioButton::indicator:checked {
                    background-color: #6366f1;
                    border: 2px solid #ffffff;
                }
            """
        else:
            # Light Mode Palette
            style = """
                QMainWindow {
                    background-color: #f8fafc;
                }
                QWidget {
                    color: #1e293b;
                    font-family: 'Inter', 'Roboto', 'Segoe UI', sans-serif;
                }
                #formCard, #resultCard {
                    background-color: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                }
                #metricSubCard {
                    background-color: #f1f5f9;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 10px;
                }
                QGroupBox {
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    margin-top: 12px;
                    padding-top: 14px;
                    font-weight: bold;
                    color: #475569;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px;
                }
                QLabel {
                    color: #1e293b;
                }
                #metaLabel {
                    color: #94a3b8;
                }
                QSpinBox, QComboBox {
                    background-color: #ffffff;
                    color: #0f172a;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    padding: 6px 10px;
                    font-size: 13px;
                }
                QSpinBox:focus, QComboBox:focus {
                    border: 1px solid #4f46e5;
                }
                QComboBox QAbstractItemView {
                    background-color: #ffffff;
                    color: #0f172a;
                    selection-background-color: #e0e7ff;
                    selection-color: #3730a3;
                    border: 1px solid #cbd5e1;
                }
                QPushButton {
                    background-color: #f8fafc;
                    color: #0f172a;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 8px 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #f1f5f9;
                    border-color: #4f46e5;
                }
                QPushButton:pressed {
                    background-color: #e2e8f0;
                }
                QPushButton[text^="⚡"] {
                    background-color: #4f46e5;
                    color: #ffffff;
                    border: none;
                }
                QPushButton[text^="⚡"]:hover {
                    background-color: #4338ca;
                }
                QProgressBar {
                    background-color: #e2e8f0;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    height: 18px;
                }
                #barGood::chunk {
                    background-color: #10b981;
                    border-radius: 5px;
                }
                #barBad::chunk {
                    background-color: #ef4444;
                    border-radius: 5px;
                }
                QRadioButton {
                    color: #334155;
                    font-weight: normal;
                }
                QRadioButton::indicator:checked {
                    background-color: #4f46e5;
                    border: 2px solid #ffffff;
                }
            """
        self.setStyleSheet(style)


def main():
    app = QApplication(sys.argv)
    window = LoanRiskStudioGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
