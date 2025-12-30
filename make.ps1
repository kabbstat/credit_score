<#
.SYNOPSIS
    MLOps automation commands for Credit Score project

.DESCRIPTION
    Run: .\make.ps1 <command>
    Example: .\make.ps1 train

.EXAMPLE
    .\make.ps1 help
#>

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$Blue = "Cyan"
$Green = "Green"
$Yellow = "Yellow"

function Write-Header {
    param([string]$Message)
    Write-Host "`n$Message" -ForegroundColor $Blue
    Write-Host ("=" * 50) -ForegroundColor $Blue
}

function Show-Help {
    Write-Host "`nCredit Score MLOps - Commands" -ForegroundColor $Blue
    Write-Host "==============================" -ForegroundColor $Blue
    Write-Host ""
    Write-Host "INSTALLATION:" -ForegroundColor $Yellow
    Write-Host "  install        " -NoNewline -ForegroundColor $Green; Write-Host "Install dependencies"
    Write-Host ""
    Write-Host "PIPELINE:" -ForegroundColor $Yellow
    Write-Host "  data-collect   " -NoNewline -ForegroundColor $Green; Write-Host "Collect raw data"
    Write-Host "  data-process   " -NoNewline -ForegroundColor $Green; Write-Host "Process data"
    Write-Host "  train          " -NoNewline -ForegroundColor $Green; Write-Host "Train model"
    Write-Host "  evaluate       " -NoNewline -ForegroundColor $Green; Write-Host "Evaluate model"
    Write-Host "  pipeline       " -NoNewline -ForegroundColor $Green; Write-Host "Run full DVC pipeline"
    Write-Host ""
    Write-Host "SERVICES:" -ForegroundColor $Yellow
    Write-Host "  mlflow         " -NoNewline -ForegroundColor $Green; Write-Host "Start MLflow UI (port 5000)"
    Write-Host "  streamlit      " -NoNewline -ForegroundColor $Green; Write-Host "Start Streamlit app (port 8501)"
    Write-Host ""
    Write-Host "DVC:" -ForegroundColor $Yellow
    Write-Host "  dvc-status     " -NoNewline -ForegroundColor $Green; Write-Host "Show pipeline status"
    Write-Host "  dvc-dag        " -NoNewline -ForegroundColor $Green; Write-Host "Show pipeline DAG"
    Write-Host "  dvc-metrics    " -NoNewline -ForegroundColor $Green; Write-Host "Show metrics"
    Write-Host ""
}

function Install-Deps {
    Write-Header "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

function Collect-Data {
    Write-Header "Collecting data..."
    python src/data_collection.py
}

function Process-Data {
    Write-Header "Processing data..."
    python src/data_prepro.py
}

function Train-Model {
    Write-Header "Training model..."
    python src/train.py
}

function Train-NoMLflow {
    Write-Header "Training model (no MLflow)..."
    python src/train.py --no-mlflow
}

function Evaluate-Model {
    Write-Header "Evaluating model..."
    python src/evaluate.py
}

function Run-Pipeline {
    Write-Header "Running DVC pipeline..."
    dvc repro
}

function Start-MLflow {
    Write-Header "Starting MLflow UI at http://localhost:5000"
    mlflow ui --host 0.0.0.0 --port 5000
}

function Start-Streamlit {
    Write-Header "Starting Streamlit app at http://localhost:8501"
    streamlit run src/eda_stream.py
}

function Show-DVCStatus { dvc status }
function Show-DVCDag { dvc dag }
function Show-DVCMetrics { dvc metrics show }

switch ($Command.ToLower()) {
    "help"          { Show-Help }
    "install"       { Install-Deps }
    "data-collect"  { Collect-Data }
    "data-process"  { Process-Data }
    "train"         { Train-Model }
    "train-simple"  { Train-NoMLflow }
    "evaluate"      { Evaluate-Model }
    "pipeline"      { Run-Pipeline }
    "mlflow"        { Start-MLflow }
    "streamlit"     { Start-Streamlit }
    "dvc-status"    { Show-DVCStatus }
    "dvc-dag"       { Show-DVCDag }
    "dvc-metrics"   { Show-DVCMetrics }
    default         { Write-Host "Unknown: $Command. Use '.\make.ps1 help'" -ForegroundColor Red }
}

