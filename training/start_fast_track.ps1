param(
    [Parameter(Mandatory=$false)]
    [string]$PythonExe = "py -3.10",

    [Parameter(Mandatory=$false)]
    [string]$SpiderLocal = "",

    [Parameter(Mandatory=$true)]
    [string]$BirdLocal,

    [Parameter(Mandatory=$false)]
    [string]$OutFile = "training/data/mixture_train.jsonl",

    [Parameter(Mandatory=$false)]
    [string]$ModelName = "google/flan-t5-base",

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "",

    [Parameter(Mandatory=$false)]
    [int]$Epochs = 2,

    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Installing dependencies..."
Invoke-Expression "$PythonExe -m pip install -r requirements.txt"

Write-Host "[2/3] Preparing mixed training dataset..."
$prepCmd = "$PythonExe training/prepare_mixture.py --bird-local `"$BirdLocal`" --out $OutFile"
if ($SpiderLocal -ne "") {
    $prepCmd += " --spider-local `"$SpiderLocal`""
}
Invoke-Expression $prepCmd

Write-Host "[3/3] Starting adapter fine-tuning..."

if ($OutputDir -eq "") {
    if ($ModelName -match "mistral") {
        $OutputDir = "training/artifacts/mistral_7b_sql_lora"
    } else {
        $OutputDir = "training/artifacts/flan_t5_sql_lora"
    }
}

if ($ModelName -match "mistral") {
    Invoke-Expression "$PythonExe training/train_mistral_qlora.py --train-file $OutFile --model $ModelName --epochs $Epochs --batch-size $BatchSize --output-dir $OutputDir"
} else {
    Invoke-Expression "$PythonExe training/train_lora.py --train-file $OutFile --model $ModelName --epochs $Epochs --batch-size $BatchSize --output-dir $OutputDir"
}

Write-Host "Completed. Model artifacts: $OutputDir/final"
