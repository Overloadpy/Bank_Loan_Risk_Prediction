#!/usr/bin/env python3
"""Command Line Interface for Bank Loan Risk Prediction System.

Built with Typer and Rich.
Author: Yordanos Andargachew (Phone: +251 952 190 305)
"""

import os
import sys
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessor import CreditDataPreprocessor
from src.predictor import LoanRiskPredictor
import pandas as pd
from sklearn.model_selection import train_test_split

app = typer.Typer(
    name="loanrisk",
    help="🏦 Production Bank Loan Risk Assessment & Inference CLI",
    add_completion=False,
)
console = Console()


def print_banner():
    """Print stylish header banner."""
    banner_text = Text()
    banner_text.append("🏦 BANK LOAN RISK PREDICTION SYSTEM\n", style="bold cyan")
    banner_text.append("Production Credit Underwriting & Decision Support CLI\n", style="italic white")
    banner_text.append("Author: Yordanos Andargachew (+251 952 190 305)", style="dim white")
    console.print(Panel(banner_text, border_style="cyan", expand=False))


@app.command()
def predict(
    age: int = typer.Option(35, "--age", "-a", help="Applicant age (years)"),
    amount: int = typer.Option(5000, "--amount", "-m", help="Requested credit amount"),
    duration: int = typer.Option(24, "--duration", "-d", help="Loan duration in months"),
    housing: str = typer.Option("own", "--housing", "-h", help="Housing status: own, rent, free"),
    saving: str = typer.Option("little", "--saving", "-s", help="Savings: little, moderate, quite rich, rich, unknown"),
    checking: str = typer.Option("moderate", "--checking", "-c", help="Checking: little, moderate, rich, unknown"),
    purpose: str = typer.Option("car", "--purpose", "-p", help="Purpose: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others"),
    sex: str = typer.Option("male", "--sex", help="Applicant biological sex: male, female"),
    job: int = typer.Option(2, "--job", "-j", help="Job category: 0 (unskilled non-resident), 1 (unskilled resident), 2 (skilled), 3 (highly skilled)"),
    model: str = typer.Option("rf", "--model", help="Model choice: rf (Random Forest) or lr (Logistic Regression)"),
):
    """Assess loan default risk for a single loan applicant."""
    print_banner()

    # Normalize inputs
    saving_val = saving.strip().lower()
    if saving_val in ["none", "nan", "null", "no", ""]:
        saving_val = "unknown"

    checking_val = checking.strip().lower()
    if checking_val in ["none", "nan", "null", "no", ""]:
        checking_val = "unknown"

    applicant_data = {
        "Age": age,
        "Sex": sex.strip().lower(),
        "Job": job,
        "Housing": housing.strip().lower(),
        "Saving accounts": saving_val,
        "Checking account": checking_val,
        "Credit amount": amount,
        "Duration": duration,
        "Purpose": purpose.strip().lower(),
    }

    try:
        predictor = LoanRiskPredictor()
        result = predictor.predict(applicant_data, model_name=model)
    except Exception as e:
        console.print(f"[bold red]❌ Error during risk assessment:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Render applicant summary table
    input_table = Table(title="📋 Applicant Profile", border_style="bright_blue", show_header=True, header_style="bold blue")
    input_table.add_column("Feature", style="cyan", justify="left")
    input_table.add_column("Value", style="bold white", justify="right")

    for k, v in applicant_data.items():
        input_table.add_row(str(k), str(v))

    console.print(input_table)

    # Verdict Card
    risk_class = result["risk_class"]
    risk_tier = result["risk_tier"]
    confidence = result["confidence_score"]
    prob_good = result["good_percentage"]
    prob_bad = result["bad_percentage"]
    rec = result["recommendation"]
    model_name = result["model_used"]

    if risk_class == "Good Risk":
        card_style = "bold green"
        badge = "🟢 [bold green]APPROVED: GOOD RISK[/bold green]"
    else:
        card_style = "bold red"
        badge = "🔴 [bold red]CAUTION: BAD RISK (HIGH DEFAULT PROPENSITY)[/bold red]"

    scorecard = Table(title="🎯 Risk Assessment Verdict", border_style=card_style, show_header=True, header_style=card_style)
    scorecard.add_column("Assessment Metric", style="bold white")
    scorecard.add_column("Outcome", style="bold yellow")

    scorecard.add_row("Decision Classification", badge)
    scorecard.add_row("Risk Tier", f"[{result['badge_color']}]{risk_tier}[/{result['badge_color']}]")
    scorecard.add_row("Model Confidence", f"[bold cyan]{confidence:.2f}%[/bold cyan]")
    scorecard.add_row("Good Repayment Probability", f"[green]{prob_good:.2f}%[/green]")
    scorecard.add_row("Default Probability", f"[red]{prob_bad:.2f}%[/red]")
    scorecard.add_row("Underwriting Recommendation", f"[italic white]{rec}[/italic white]")
    scorecard.add_row("Scoring Engine Used", f"[dim cyan]{model_name}[/dim cyan]")

    console.print(scorecard)


@app.command()
def evaluate(
    split: str = typer.Option("test", "--split", help="Dataset split to evaluate: test or full"),
):
    """Evaluate and compare baseline vs advanced model metrics on holdout data."""
    print_banner()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset.csv")
    if not os.path.exists(data_path):
        console.print(f"[bold red]❌ Error: Dataset file '{data_path}' not found.[/bold red]")
        raise typer.Exit(code=1)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["Risk"])
    y = CreditDataPreprocessor.encode_target(df["Risk"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    if split.lower() == "full":
        eval_X = X
        eval_y = y
        split_desc = "Full Dataset (1,000 samples)"
    else:
        eval_X = X_test
        eval_y = y_test
        split_desc = "20% Stratified Holdout Test Set (200 samples)"

    try:
        predictor = LoanRiskPredictor()
        X_trans = predictor.preprocessor.transform(eval_X)
        metrics = predictor.evaluate(X_trans, eval_y)
    except Exception as e:
        console.print(f"[bold red]❌ Evaluation error:[/bold red] {e}")
        raise typer.Exit(code=1)

    table = Table(
        title=f"📊 Model Performance Comparison ({split_desc})",
        border_style="magenta",
        header_style="bold magenta",
    )
    table.add_column("Performance Metric", style="cyan", justify="left")
    table.add_column("Logistic Regression (Baseline)", style="bold yellow", justify="center")
    table.add_column("Random Forest (Advanced)", style="bold green", justify="center")
    table.add_column("Delta (RF vs LR)", style="bold white", justify="center")

    lr_m = metrics["logistic_regression"]
    rf_m = metrics["random_forest"]

    metric_names = [
        ("Accuracy", "accuracy"),
        ("Precision (Bad Risk)", "precision"),
        ("Recall (Bad Risk)", "recall"),
        ("F1-Score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
    ]

    for label, key in metric_names:
        lr_val = lr_m[key]
        rf_val = rf_m[key]
        delta = rf_val - lr_val
        delta_str = f"[green]+{delta:.4f}[/green]" if delta > 0 else f"[red]{delta:.4f}[/red]"
        table.add_row(label, f"{lr_val:.4f}", f"{rf_val:.4f}", delta_str)

    console.print(table)

    # Confusion matrix breakdown
    cm_table = Table(title="🔍 Confusion Matrix Breakdown", border_style="cyan", header_style="bold cyan")
    cm_table.add_column("Model", style="bold white")
    cm_table.add_column("True Good (TN)", style="green", justify="center")
    cm_table.add_column("False Bad (FP)", style="yellow", justify="center")
    cm_table.add_column("False Good (FN - Costly Risk)", style="red", justify="center")
    cm_table.add_column("True Bad (TP)", style="green", justify="center")

    lr_cm = lr_m["confusion_matrix"]
    rf_cm = rf_m["confusion_matrix"]

    cm_table.add_row("Logistic Regression", str(lr_cm[0][0]), str(lr_cm[0][1]), str(lr_cm[1][0]), str(lr_cm[1][1]))
    cm_table.add_row("Random Forest", str(rf_cm[0][0]), str(rf_cm[0][1]), str(rf_cm[1][0]), str(rf_cm[1][1]))

    console.print(cm_table)


if __name__ == "__main__":
    app()
