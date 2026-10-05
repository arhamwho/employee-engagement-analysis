"""
Shared settings and folder paths for this project.

These constants keep every script and notebook pointed at the same
dataset file. Change the filename here only — do not scatter path
strings across the project.
"""

from pathlib import Path

# Project root is one folder above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
EDA_PLOTS_DIR = REPORTS_DIR / "eda_plots"
TARGET_PLOTS_DIR = REPORTS_DIR / "target_plots"
MODEL_PLOTS_DIR = REPORTS_DIR / "model_plots"

# Reproducible split used by every classifier.
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Dataset file
# ---------------------------------------------------------------------------
# Source family: IBM HR Analytics Employee Attrition & Performance.
# The file currently in this project is an enhanced copy of that table.
EXPECTED_CSV_FILENAME = "IBM_HR_Attrition_Enhanced.csv"
EXPECTED_CSV_PATH = DATA_DIR / EXPECTED_CSV_FILENAME

# Other names this loader will accept if you swap files later.
ALTERNATE_CSV_FILENAMES = (
    "ibm_hr_employee_attrition.csv",
    "WA_Fn-UseC_-HR-Employee-Attrition.csv",
)

CLUSTERING_REPORTS_DIR = REPORTS_DIR / "clustering"
CLUSTERING_PLOTS_DIR = CLUSTERING_REPORTS_DIR / "plots"
CLASSIFICATION_BASELINE_DIR = REPORTS_DIR / "classification_baseline"

CLUSTERING_PIPELINE_FILENAME = "engagement_clustering_pipeline.joblib"
CLUSTERING_PIPELINE_PATH = MODELS_DIR / CLUSTERING_PIPELINE_FILENAME

SURVEY_DATA_DIR = DATA_DIR / "survey"
SURVEY_MODELS_DIR = MODELS_DIR / "survey"
SURVEY_REPORTS_DIR = REPORTS_DIR / "survey_model"
SURVEY_PLOTS_DIR = SURVEY_REPORTS_DIR / "plots"

SURVEY_TEMPLATE_FILENAME = "employee_engagement_survey_template.csv"
SURVEY_DATA_FILENAME = "employee_engagement_survey.csv"
SURVEY_TEMPLATE_PATH = SURVEY_DATA_DIR / SURVEY_TEMPLATE_FILENAME
SURVEY_DATA_PATH = SURVEY_DATA_DIR / SURVEY_DATA_FILENAME
