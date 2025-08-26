# ============================================================
# ETTm2 multivariate forecaster (PyTorch + PennyLane)
# - Multivariate [B, pred_len, F]
# - He (Kaiming) init + LayerNorm backbone
# - Learnable quantum-gated sigmoid (clamped)  -> g ~ 0.5 la start
# - Residual to last observed step (y = base + last_obs)
# - Early stopping + ReduceLROnPlateau + weight decay
# - Reports ONLY MAE & MSE on NORMALIZED data
# ============================================================

import os, urllib.request, numpy as np, pandas as pd, random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------- Config --------------------
DATA_DIR   = "./data/ETT-small"
CSV_URL    = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"
CSV_PATH   = os.path.join(DATA_DIR, "ETTm2.csv")

seed       = 42
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len    = 96
pred_len   = 720
batch_size = 256
epochs     = 120
lr         = 1e-3
patience   = 10

# model dims
k          = 128
d          = 128            # ↑ capacitate
drop_prob  = 0.1
weight_decay = 1e-4

print("Device:", device)

# -------------------- Repro --------------------
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------------- Download dataset --------------------
def ensure_ettm2(csv_path=CSV_PATH, url=CSV_URL):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        print("Downloading ETTm2.csv ...")
        urllib.request.urlretrieve(url, csv_path)
        print("Saved to", csv_path)
    else:
        print("ETTm2.csv already present at", csv_path)

ensure_ettm2()

# -------------------- Load + calendar feats + normalization --------------------
df = pd.read_csv(CSV_PATH)

dt = pd.to_datetime(df['date'])
df['hour']  = dt.dt.hour + dt.dt.minute/60.0
df['day']   = dt.dt.dayofyear
df['sin_h'] = np.sin(2*np.pi*df['hour']/24.0)
df['cos_h'] = np.cos(2*np.pi*df['hour']/24.0)
df['sin_d'] = np.sin(2*np.pi*df['day']/365.0)
df['cos_d'] = np.cos(2*np.pi*df['day']/365.0)

feature_cols = [c for c in df.columns if c != 'date']
X_all = df[feature_cols].values.astype(np.float32)
col_names = feature_cols[:]

N = len(df)
n_train = int(0.6*N); n_val = int(0.2*N)
idx_train = (0, n_train); idx_val = (n_train, n_train+n_val); idx_test = (n_train+n_val, N)

scaler = StandardScaler()
X_all[:n_train] = scaler.fit_transform(X_all[:n_train])
X_all[n_train:] = scaler.transform(X_all[n_train:])

n_features = X_all.shape[1]

# -------------------- Dataset --------------------
class ETTWindowed(Dataset):
    def __init__(self, X, start, end, seq_len, pred_len):
        self.X = X
        self.first = start
        self.last  = end - (seq_len + pred_len)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.len   = max(0, self.last - self.first + 1)
    def __len__(self): return self.len
    def __getitem__(self, i):
        s = self.first + i; e = s + self.seq_len; f = e + self.pred_len
        seq = self.X[s:e]        # [T,F]
        tgt = self.X[e:f]        # [H,F]
        return torch.from_numpy(seq), torch.from_numpy(tgt)

ds_train = ETTWindowed(X_all, *idx_train, seq_len, pred_len)
ds_val   = ETTWindowed(X_all, *idx_val,   seq_len, pred_len)
ds_test  = ETTWindowed(X_all, *idx_test,  seq_len, pred_len)

dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  drop_last=True)
dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)
dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, drop_last=False)

print(f"Splits -> train:{len(ds_train)}  val:{len(ds_val)}  test:{len(ds_test)}  | feats:{n_features}")

# -------------------- Quantum gate --------------------
dev_q = qml.device("default.qubit", wires=1)

@qml.qnode(dev_q, interface="torch")
def qubit_z(theta, phi):
    qml.RY(theta, wires=0)
    qml.RX(phi,   wires=0)
    return qml.expval(qml.PauliZ(0))   # z in [-1,1]

def kaiming_normal_all_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# -------------------- Model --------------------
class QTemporal(nn.Module):
    """
    h_t = (1-g)*h_{t-1} + g * LN(W(P(x_t)) + b + alpha * cal_scalar)
    g = sigmoid(w1*z1 + w2*z2 + b_g), z1,z2 = <Z> qubit; g clamped [g_min, g_max]
    """
    def __init__(self, n_features, k=128, d=128, cal_dims=4, g_min=0.05, g_max=0.95):
        super().__init__()
        # quantum params
        self.theta1 = nn.Parameter(torch.tensor(0.2))
        self.phi1   = nn.Parameter(torch.tensor(0.1))
        self.theta2 = nn.Parameter(torch.tensor(0.3))
        self.phi2   = nn.Parameter(torch.tensor(0.05))
        # learnable combiner for gate
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.b_g = nn.Parameter(torch.tensor(0.0))
        self.g_min = g_min
        self.g_max = g_max

        # backbone
        self.P  = nn.Linear(n_features, k, bias=False)
        self.W  = nn.Linear(k, d)
        self.b  = nn.Parameter(torch.zeros(d))
        self.ln = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.d = d
        self.cal_dims = cal_dims
        kaiming_normal_all_linear(self)

    def gate(self):
        z1 = qubit_z(self.theta1, self.phi1)
        z2 = qubit_z(self.theta2, self.phi2)
        g  = torch.sigmoid(self.w1*z1 + self.w2*z2 + self.b_g)  # [scalar]
        # clamp pentru a evita „poarta moartă”
        g  = torch.clamp(g, self.g_min, self.g_max)
        return g

    def forward(self, x):  # x: [B,T,F]
        B, T, F = x.shape
        h = torch.zeros(B, self.d, device=x.device)
        g_val = self.gate()

        if F >= self.cal_dims:
            cal = x[..., -self.cal_dims:]          # [B,T,4]
            cal_aggr = cal.mean(dim=1)             # [B,4]
        else:
            cal_aggr = torch.zeros(B, self.cal_dims, device=x.device)
        cal_scalar = cal_aggr.mean(dim=1, keepdim=True)  # [B,1]

        for t in range(T):
            xt   = x[:, t, :]                    # [B,F]
            base = self.W(self.P(xt)) + self.b   # [B,d]
            base = base + self.alpha * cal_scalar
            base = self.ln(base)
            h    = (1 - g_val) * h + g_val * base
        return h  # [B,d]

class QETTForecaster(nn.Module):
    def __init__(self, n_features, k=128, d=128, pred_len=24, drop_prob=0.1):
        super().__init__()
        self.backbone = QTemporal(n_features, k=k, d=d, cal_dims=4)
        self.dec = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(d, pred_len * n_features)
        )
        self.pred_len = pred_len
        self.n_features = n_features
        kaiming_normal_all_linear(self)

    def forward(self, x):  # [B,T,F]
        h = self.backbone(x)                      # [B,d]
        y_base = self.dec(h)                      # [B, pred_len*n_features]
        y_base = y_base.view(x.size(0), self.pred_len, self.n_features)  # [B,H,F]
        # ===== residual to last observed value =====
        last_obs = x[:, -1, :].unsqueeze(1)      # [B,1,F]
        y = y_base + last_obs                     # predict deviations
        return y

model = QETTForecaster(n_features=n_features, k=k, d=d, pred_len=pred_len, drop_prob=drop_prob).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)
lossf = nn.MSELoss()

print("Params:", sum(p.numel() for p in model.parameters()))

# -------------------- Train + Early Stopping --------------------
best_val = float('inf'); best_state = None; no_improve = 0

for ep in range(1, epochs+1):
    model.train()
    tr_sum, n_tr = 0.0, 0
    for xb, yb in dl_train:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        opt.zero_grad()
        pred = model(xb)
        loss = lossf(pred, yb)
        loss.backward()
        opt.step()
        tr_sum += loss.item() * xb.size(0); n_tr += xb.size(0)
    tr_mse = tr_sum / max(1, n_tr)

    model.eval()
    va_sum, n_va = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(device).float(); yb = yb.to(device).float()
            pred = model(xb)
            va_sum += lossf(pred, yb).item() * xb.size(0); n_va += xb.size(0)
    va_mse = va_sum / max(1, n_va)
    sched.step(va_mse)

    g_now = model.backbone.gate().item()
    print(f"Epoch {ep:03d} | train MSE={tr_mse:.6f} | val MSE={va_mse:.6f} | g={g_now:.3f} | lr={opt.param_groups[0]['lr']:.1e}")

    if va_mse + 1e-8 < best_val:
        best_val = va_mse
        best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {ep} (no improvement {no_improve}/{patience}).")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# -------------------- Test + MAE/MSE on NORMALIZED data --------------------
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in dl_test:
        xb = xb.to(device).float()
        pred = model(xb).cpu().numpy()
        preds.append(pred); trues.append(yb.numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

MAE_all = mean_absolute_error(trues.reshape(-1, n_features),
                              preds.reshape(-1, n_features))
MSE_all = mean_squared_error(trues.reshape(-1, n_features),
                             preds.reshape(-1, n_features))
print("\n───────── TEST METRICS (Normalized) ─────────")
print(f"MAE : {MAE_all:.6f}")
print(f"MSE : {MSE_all:.6f}")
