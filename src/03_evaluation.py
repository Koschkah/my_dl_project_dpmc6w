#!/usr/bin/env python
"""
03_evaluation.py

GNN (GraphSAGE) modell kiértékelése + baseline eredmények összevonása.

Bemenet:
    /app/output/processed_dataset.csv
    /app/output/models/gnn_sage_k2.pt
    /app/output/baselines/gnn_x_scaler.joblib
    /app/output/baselines/gnn_y_scaler.joblib
    /app/output/baselines/baseline_results.csv (02_train hozza létre)
    /data/stop_times.txt

Kimenet:
    /app/output/evaluation/baseline_evaluation.csv
    /app/output/evaluation/baseline_evaluation.json
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# GLOBÁLIS KONSTANSOK
# ============================================================

DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/app/output")
BASELINE_DIR = OUTPUT_DIR / "baselines"
MODELS_DIR = OUTPUT_DIR / "models"

FEATURES = ["delay_seconds_calc", "hour", "weekday"]
TARGET = "y_end_delay_calc"
STOP_COL = "last_stop_id"
K_HOP = 2


# ============================================================
# 1. GNN MODEL (SimpleSAGE)
# ============================================================
class SimpleSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch, center_pos):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)

        num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 0
        counts = torch.bincount(batch, minlength=num_graphs)
        offsets = torch.cumsum(counts, dim=0) - counts
        center_global = offsets + center_pos.view(-1)

        x_center = x[center_global]
        return self.lin(x_center).squeeze(-1)


# ============================================================
# 2. STOP GRAPH építése (ugyanaz logika, mint 02_train-ben)
# ============================================================
def build_stop_graph(stop_times_path: Path):
    st = pd.read_csv(stop_times_path, dtype={"stop_id": str})
    st = st[["trip_id", "stop_id", "stop_sequence"]]
    st = st.sort_values(["trip_id", "stop_sequence"])

    unique_stops = pd.Index(st["stop_id"].unique())
    stop_id_to_idx = {s: i for i, s in enumerate(unique_stops)}
    num_nodes = len(unique_stops)

    edges_src, edges_dst = [], []
    for _, group in st.groupby("trip_id"):
        stops = group["stop_id"].astype(str).values
        for i in range(len(stops) - 1):
            u = stop_id_to_idx[stops[i]]
            v = stop_id_to_idx[stops[i + 1]]
            edges_src.append(u)
            edges_dst.append(v)
            edges_src.append(v)
            edges_dst.append(u)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return stop_id_to_idx, edge_index, num_nodes


# ============================================================
# 3. K-HOP CACHE csak használt node-okra
# ============================================================
def build_khop_cache(edge_index, num_nodes, k, used_nodes):
    cache = {}
    for center in used_nodes:
        nodes, edge_index_sub, _, _ = k_hop_subgraph(
            center, k, edge_index, relabel_nodes=True
        )
        center_pos = (nodes == center).nonzero(as_tuple=True)[0]
        center_pos = int(center_pos[0]) if len(center_pos) > 0 else 0
        cache[center] = (nodes, edge_index_sub, center_pos)
    return cache


# ============================================================
# 4. Row-level PyG Dataset
# ============================================================
class RowGNNDataset(Dataset):
    def __init__(
        self,
        df,
        stop_id_to_idx,
        khop_cache,
        x_scaler,
        y_scaler,
        features,
        target,
        stop_col,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.stop_id_to_idx = stop_id_to_idx
        self.khop_cache = khop_cache
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.features = features
        self.target = target
        self.stop_col = stop_col

        # node feature dim: input features + center_flag
        self.in_dim = len(features) + 1

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]

        # -------- center stop --------
        stop_id = str(row[self.stop_col])
        center_idx = self.stop_id_to_idx.get(stop_id, None)

        # -------- target --------
        y_scaled = self.y_scaler.transform(
            [[row[self.target]]]
        )[0, 0]

        if center_idx is None:
            x = torch.zeros((1, self.in_dim), dtype=torch.float32)

            edge_index = torch.tensor(
                [[0], [0]], dtype=torch.long
            )

            return Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([y_scaled], dtype=torch.float32),
                center_pos=torch.tensor([0], dtype=torch.long),
            )


        nodes, edge_index_sub, center_pos = self.khop_cache[center_idx]

        # ---- center node feature scaling ----
        x_raw = (
            row[self.features]
            .values.astype(np.float32)
            .reshape(1, -1)
        )
        x_scaled = self.x_scaler.transform(x_raw)[0]

        # ---- node feature mátrix ----
        # minden node 0, csak center kap feature + flag
        x_sub = np.zeros(
            (nodes.shape[0], self.in_dim),
            dtype=np.float32,
        )
        x_sub[center_pos, : len(self.features)] = x_scaled
        x_sub[center_pos, -1] = 1.0  # center_flag

        return Data(
            x=torch.from_numpy(x_sub),
            edge_index=edge_index_sub,
            y=torch.tensor([y_scaled], dtype=torch.float32),
            center_pos=torch.tensor([center_pos], dtype=torch.long),
        )


# ============================================================
# 5. METRIKÁK
# ============================================================
def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
    }

def threshold_accuracy(y_true: np.ndarray, y_pred: np.ndarray, thresholds=(60, 120, 180)):
    """Accuracy: arány, ahol |pred-true| <= threshold."""
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    abs_err = np.abs(y_pred - y_true)
    out = {}
    for t in thresholds:
        out[f"ACC@{t}s"] = float((abs_err <= t).mean())
    return out


def format_acc_table(acc_dict_by_model: dict) -> pd.DataFrame:
    """acc_dict_by_model: {model_name: {"ACC@60s":..., ...}} -> DataFrame"""
    df = pd.DataFrame.from_dict(acc_dict_by_model, orient="index")
    df.index.name = "model"
    # rendezés oszlopok szerint, ha kell
    cols = [c for c in ["ACC@60s", "ACC@120s", "ACC@180s"] if c in df.columns]
    return df[cols].reset_index()


# ============================================================
# 6. MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True) 

    # ----- Dataset beolvasása + split (ugyanúgy mint 02-ben) -----
    data_path = DATA_DIR / "processed_dataset.csv"
    df = pd.read_csv(data_path, parse_dates=["vehicle_timestamp"])
    df = df.sort_values("vehicle_timestamp")
    df[STOP_COL] = df[STOP_COL].astype(str)
    df = df.dropna(subset=FEATURES + [TARGET, STOP_COL]).copy()

    MAX_ITER_ROWS = 100000
    if MAX_ITER_ROWS != -1:
        df = df.head(MAX_ITER_ROWS)
        print(f"[03_evaluation] Limiting to first {MAX_ITER_ROWS} rows for faster iteration.")
    else:
        print(f"[03_evaluation] Using all {len(df)} rows.")

    train_end = df["vehicle_timestamp"].quantile(0.70)
    val_end = df["vehicle_timestamp"].quantile(0.85)

    train_df = df[df["vehicle_timestamp"] < train_end].copy()
    val_df = df[(df["vehicle_timestamp"] >= train_end) & (df["vehicle_timestamp"] < val_end)].copy()
    test_df = df[df["vehicle_timestamp"] >= val_end].copy()

    print("[03_evaluation] rows:", len(df))
    print("[03_evaluation] train/val/test:", len(train_df), len(val_df), len(test_df))


    # ----- Baseline predikciók a test halmazon (accuracy-hoz is kell) -----
    y_true_test = test_df[TARGET].astype(float).values

    # Mean baseline: train átlag (ugyanaz split logikával, mint 02)
    mean_target = train_df[TARGET].astype(float).mean()
    pred_mean = np.full_like(y_true_test, fill_value=mean_target, dtype=float)

    # Current delay baseline
    pred_current = test_df["delay_seconds_calc"].astype(float).values

    # Linear Regression baseline betöltése + pred
    lr_path = BASELINE_DIR / "linear_regression.joblib"
    lr_scaler_path = BASELINE_DIR / "linear_regression_scaler.joblib"

    pred_lr = None
    if lr_path.exists() and lr_scaler_path.exists():
        lr = joblib.load(lr_path)
        lr_scaler = joblib.load(lr_scaler_path)
        X_test_lr = test_df[FEATURES].values
        X_test_lr_s = lr_scaler.transform(X_test_lr)
        pred_lr = lr.predict(X_test_lr_s).astype(float)
    else:
        print("[03_evaluation][WARN] LinearRegression artifacts missing, skipping LR accuracy.")





    # ----- GNN modell + scaler-ek betöltése -----
    ckpt_path = MODELS_DIR / "gnn_sage_k2.pt"
    x_scaler_path = BASELINE_DIR / "gnn_x_scaler.joblib"
    y_scaler_path = BASELINE_DIR / "gnn_y_scaler.joblib"

    if not (ckpt_path.exists() and x_scaler_path.exists() and y_scaler_path.exists()):
        print("[03_evaluation] Missing GNN checkpoint or scalers, aborting GNN eval.")
        return

    print("[03_evaluation] Loading GNN model and scalers...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_channels = ckpt.get("in_channels", len(FEATURES) + 1)
    hidden_channels = ckpt.get("hidden_channels", 64)
    num_layers = ckpt.get("num_layers", 2)
    k_hop = ckpt.get("k_hop", K_HOP)

    model = SimpleSAGE(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    # ----- Gráf + k-hop cache újraépítése -----
    print("[03_evaluation] Rebuilding stop graph...")
    stop_times_path = DATA_DIR / "stop_times.txt"
    stop_id_to_idx, edge_index, num_nodes = build_stop_graph(stop_times_path)

    used_stop_ids = test_df[STOP_COL].astype(str).unique()
    used_nodes = [stop_id_to_idx[s] for s in used_stop_ids if s in stop_id_to_idx]
    print(f"[03_evaluation] Used stops in test: {len(used_nodes)} / {num_nodes}")

    khop_cache = build_khop_cache(edge_index, num_nodes, k_hop, used_nodes)

    # ----- Dataset + DataLoader -----
    gnn_test_ds = RowGNNDataset(
    test_df,
    stop_id_to_idx,
    khop_cache,
    x_scaler,
    y_scaler,
    FEATURES,
    TARGET,
    STOP_COL,
)
    test_loader = DataLoader(gnn_test_ds, batch_size=128, shuffle=False)

    # ----- Inference -----
    print("[03_evaluation] Running GNN inference on test set...")
    preds_scaled = []
    y_scaled_true = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, batch.center_pos)

            preds_scaled.append(out.cpu().numpy())
            y_scaled_true.append(batch.y.view(-1).cpu().numpy())

    preds_scaled = np.concatenate(preds_scaled, axis=0)
    y_scaled_true = np.concatenate(y_scaled_true, axis=0)

    # visszaskálázás
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_scaled_true.reshape(-1, 1)).ravel()

    gnn_metrics = eval_metrics(y_true, preds)

    # ----- Threshold accuracy (60/120/180s) -----
    acc_by_model = {}

    acc_by_model["baseline_mean"] = threshold_accuracy(y_true_test, pred_mean, thresholds=(60, 120, 180))
    acc_by_model["baseline_current_delay"] = threshold_accuracy(y_true_test, pred_current, thresholds=(60, 120, 180))

    if pred_lr is not None:
        acc_by_model["linear_regression"] = threshold_accuracy(y_true_test, pred_lr, thresholds=(60, 120, 180))

    # GNN: itt y_true a test cél, preds a gnn pred (mindkettő másodpercben)
    acc_by_model["gnn_sage_k2"] = threshold_accuracy(y_true, preds, thresholds=(60, 120, 180))

    acc_table = format_acc_table(acc_by_model)



    print(
        f"[03_evaluation] GNN (GraphSAGE K={k_hop}) TEST "
        f"MAE={gnn_metrics['MAE']:.2f}  RMSE={gnn_metrics['RMSE']:.2f}"
    )

    # ----- Baseline eredmények betöltése + bővítése -----
    baseline_csv_path = BASELINE_DIR / "baseline_results.csv"
    if baseline_csv_path.exists():
        base_res = pd.read_csv(baseline_csv_path)
    else:
        base_res = pd.DataFrame(columns=["model", "model_type", "MAE_seconds", "RMSE_seconds"])

    new_row = {
        "model": "gnn_sage_k2",
        "model_type": "gnn",
        "MAE_seconds": gnn_metrics["MAE"],
        "RMSE_seconds": gnn_metrics["RMSE"],
    }

    if not base_res.empty and "model" in base_res.columns:
        mask = base_res["model"] == "gnn_sage_k2"
        if mask.any():
            base_res.loc[mask, ["MAE_seconds", "RMSE_seconds"]] = new_row["MAE_seconds"], new_row["RMSE_seconds"]
        else:
            base_res = pd.concat([base_res, pd.DataFrame([new_row])], ignore_index=True)
    else:
        base_res = pd.DataFrame([new_row])

    # ----- Kiértékelés mentése -----
    eval_out_dir = OUTPUT_DIR / "evaluation"
    eval_out_dir.mkdir(exist_ok=True)

    eval_csv = eval_out_dir / "baseline_evaluation.csv"
    eval_json = eval_out_dir / "baseline_evaluation.json"

    base_res.to_csv(eval_csv, index=False)
    base_res.to_json(eval_json, orient="records", indent=2)
    
    acc_csv = eval_out_dir / "threshold_accuracy.csv"
    acc_table.to_csv(acc_csv, index=False)
    print(f"[03_evaluation] Saved threshold accuracy to {acc_csv}")

    print(f"[03_evaluation] Saved baseline+GNN eval to {eval_csv}")
    print(f"[03_evaluation] Saved baseline+GNN eval JSON to {eval_json}")


    #print the evaluatoin metrics to log

    print("[03_evaluation] baseline_evaluation.csv content:")
    print(base_res.to_string(index=False))
    print()
    
    print("\n[03_evaluation] Threshold accuracy table (|pred-true| <= T):")
    print(acc_table.to_string(index=False))
    print()




if __name__ == "__main__":
    main()
