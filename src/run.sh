#!/usr/bin/env bash
set -euo pipefail

echo "[run] Starting pipeline..."

# ---- Paths (container conventions) ----
DATA_DIR="/data"
OUT_DIR="/app/output"

# Where are the scripts?
# - Some setups copy scripts to /app
# - Your current container runs from /workspace/src
if [[ -f "/workspace/src/01_data_processing.py" ]]; then
  SRC_DIR="/workspace/src"
elif [[ -f "/app/01_data_processing.py" ]]; then
  SRC_DIR="/app"
else
  echo "[run][ERROR] Cannot find 01_data_processing.py in /workspace/src or /app"
  exit 1
fi

# ---- Basic checks ----
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[run][ERROR] Missing $DATA_DIR (did you mount the host data folder to /data?)"
  exit 1
fi

# ---- Required inputs (download if missing) ----
REQUIRED_FILES=("vehicle_updates.csv" "stop_times.txt" "trips.txt")

missing_any=false
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$DATA_DIR/$f" ]]; then
    echo "[run][WARN] Missing input file: $DATA_DIR/$f"
    missing_any=true
  fi
done

if [[ "$missing_any" == true ]]; then
  echo "[run] Some input files are missing. Running data_download.py..."

  if [[ -f "$SRC_DIR/data_download.py" ]]; then
    python "$SRC_DIR/data_download.py"
  elif [[ -f "$SRC_DIR/tools/data_download.py" ]]; then
    python "$SRC_DIR/tools/data_download.py"
  else
    echo "[run][ERROR] data_download.py not found in $SRC_DIR or $SRC_DIR/tools"
    exit 1
  fi

  echo "[run] Re-checking required input files after download..."
  for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$DATA_DIR/$f" ]]; then
      echo "[run][ERROR] Still missing after download: $DATA_DIR/$f"
      exit 1
    fi
  done
fi


mkdir -p "$OUT_DIR"

echo "[run] Using scripts from: $SRC_DIR"
echo "[run] Data dir: $DATA_DIR"
echo "[run] Output dir: $OUT_DIR"

# ---- Step 1: data processing ----
echo "[run] (1/3) Data processing..."
python "$SRC_DIR/01_data_processing.py"

# sanity check output
if [[ ! -f "$DATA_DIR/processed_dataset.csv" ]]; then
  echo "[run][ERROR] 01_data_processing did not produce $DATA_DIR/processed_dataset.csv"
  exit 1
fi

# ---- Step 2: training ----
echo "[run] (2/3) Training..."
python "$SRC_DIR/02_train.py"

# sanity check key artifacts (GNN + scalers)
if [[ ! -f "$OUT_DIR/models/gnn_sage_k2.pt" ]]; then
  echo "[run][ERROR] Missing GNN model artifact: $OUT_DIR/models/gnn_sage_k2.pt"
  exit 1
fi
if [[ ! -f "$OUT_DIR/baselines/gnn_x_scaler.joblib" ]] || [[ ! -f "$OUT_DIR/baselines/gnn_y_scaler.joblib" ]]; then
  echo "[run][ERROR] Missing GNN scaler artifacts in $OUT_DIR/baselines"
  exit 1
fi

# ---- Step 3: evaluation ----
echo "[run] (3/3) Evaluation..."
python "$SRC_DIR/03_evaluation.py"

# sanity check evaluation outputs
if [[ ! -f "$OUT_DIR/evaluation/baseline_evaluation.csv" ]]; then
  echo "[run][ERROR] Missing evaluation output: $OUT_DIR/evaluation/baseline_evaluation.csv"
  exit 1
fi

echo "[run] Pipeline finished successfully."
echo "[run] Outputs:"
echo "  - $OUT_DIR/processed_dataset.csv"
echo "  - $OUT_DIR/baselines/"
echo "  - $OUT_DIR/models/"
echo "  - $OUT_DIR/evaluation/"
