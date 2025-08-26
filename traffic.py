# ============================================================
# Traffic @ H=720 — same architecture as ETT (QTemporal + MLP + residual)
# Fixes: predict ONLY sensors (exclude 8 calendar feats from targets),
#        per-sensor MinMax scaling, smaller d, stronger gate clamp,
#        gradient clipping, loss/metrics on sensors only.
# ============================================================

import os, urllib.request, numpy as np, pandas as pd, random, time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------- Config --------------------
DATA_DIR   = "./data/traffic"
URL_GZ     = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz"
PATH_GZ    = os.path.join(DATA_DIR, "traffic.txt.gz")

seed       = 42
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len    = 96          # poți încerca și 192/168/96
pred_len   = 720          # ținta principală
batch_size = 128
epochs     = 120
lr         = 1e-3
patience   = 12
weight_decay = 1e-4
grad_clip  = 1.0

# model dims (aceeași topologie, capacitate mai mică)
k          = 128
d          = 64           # ↓ din 128 => regularizare implicită
drop_prob  = 0.2
g_min, g_max = 0.05, 0.75 # clamp mai strict pentru gate

print("Device:", device)

# -------------------- Repro --------------------
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------------- Download dataset --------------------
def ensure_traffic(path_gz=PATH_GZ, url=URL_GZ):
    os.makedirs(os.path.dirname(path_gz), exist_ok=True)
    if not os.path.exists(path_gz):
        print("Downloading traffic.txt.gz ...")
        urllib.request.urlretrieve(url, path_gz)
        print("Saved to", path_gz)
    else:
        print("traffic.txt.gz already present at", path_gz)

ensure_traffic()

# -------------------- Load + seasonal feats --------------------
df_raw = pd.read_csv(PATH_GZ, header=None, compression='gzip')
values = df_raw.values.astype(np.float32)   # [T, F_sensors]
T_total, F_sensors = values.shape

idx = pd.date_range("2015-01-01 00:00:00", periods=T_total, freq="h")
hour = idx.hour.to_numpy()
dow  = idx.dayofweek.to_numpy()
doy  = idx.dayofyear.to_numpy()

sin_h   = np.sin(2*np.pi*hour/24.0).astype(np.float32)
cos_h   = np.cos(2*np.pi*hour/24.0).astype(np.float32)
sin_dow = np.sin(2*np.pi*dow/7.0).astype(np.float32)
cos_dow = np.cos(2*np.pi*dow/7.0).astype(np.float32)
sin_doy = np.sin(2*np.pi*doy/365.0).astype(np.float32)
cos_doy = np.cos(2*np.pi*doy/365.0).astype(np.float32)
is_weekend = ((dow == 5) | (dow == 6)).astype(np.float32)
is_rush    = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))).astype(np.float32)

CAL_DIMS = 8
cal_feats = np.stack([sin_h,cos_h,sin_dow,cos_dow,sin_doy,cos_doy,is_weekend,is_rush], axis=1)  # [T,8]

# Concatenăm pentru input: [senzori | calendar]
X_all = np.concatenate([values, cal_feats], axis=1).astype(np.float32)  # [T, F_sensors+8]
F_in  = X_all.shape[1]
print(f"Raw sensors: {F_sensors} | total feats (with calendar): {F_in}")

# -------------------- Split 60/20/20 + scaling --------------------
N = X_all.shape[0]
n_train = int(0.6*N); n_val = int(0.2*N)
idx_train = (0, n_train); idx_val = (n_train, n_train+n_val); idx_test = (n_train+n_val, N)

# MinMax per-coloană (senzori + calendar) pe train, apoi transform pe val/test
scaler = MinMaxScaler()
X_all[:idx_train[1]] = scaler.fit_transform(X_all[:idx_train[1]])
X_all[idx_train[1]:] = scaler.transform(X_all[idx_train[1]:])

# -------------------- Dataset (targets = SENSORS ONLY) --------------------
class WindowedDS(Dataset):
    def __init__(self, X, start, end, seq_len, pred_len, f_sensors, cal_dims):
        self.X = X
        self.first = start
        self.last  = end - (seq_len + pred_len)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.len   = max(0, self.last - self.first + 1)
        self.f_sensors = f_sensors
        self.cal_dims = cal_dims
    def __len__(self): return self.len
    def __getitem__(self, i):
        s = self.first + i; e = s + self.seq_len; f = e + self.pred_len
        seq = self.X[s:e]                                  # [T, F_in = sensors+8]
        tgt = self.X[e:f, :self.f_sensors]                 # [H, F_sensors]  <-- DOAR senzori
        return torch.from_numpy(seq), torch.from_numpy(tgt)

ds_train = WindowedDS(X_all, *idx_train, seq_len, pred_len, F_sensors, CAL_DIMS)
ds_val   = WindowedDS(X_all, *idx_val,   seq_len, pred_len, F_sensors, CAL_DIMS)
ds_test  = WindowedDS(X_all, *idx_test,  seq_len, pred_len, F_sensors, CAL_DIMS)

dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  drop_last=True)
dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)
dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, drop_last=False)

# -------------------- Model (identic ca ETT, dar decoder = H x F_sensors) --------------------
dev_q = qml.device("default.qubit", wires=1)
@qml.qnode(dev_q, interface="torch")
def qubit_z(theta, phi):
    qml.RY(theta, wires=0); qml.RX(phi, wires=0)
    return qml.expval(qml.PauliZ(0))

def kaiming_normal_all_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class QTemporal(nn.Module):
    def __init__(self, n_features_in, k=128, d=64, cal_dims=8, g_min=0.05, g_max=0.75):
        super().__init__()
        self.theta1 = nn.Parameter(torch.tensor(0.2))
        self.phi1   = nn.Parameter(torch.tensor(0.1))
        self.theta2 = nn.Parameter(torch.tensor(0.3))
        self.phi2   = nn.Parameter(torch.tensor(0.05))
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.b_g = nn.Parameter(torch.tensor(0.0))
        self.g_min, self.g_max = g_min, g_max

        self.P  = nn.Linear(n_features_in, k, bias=False)
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
        g  = torch.sigmoid(self.w1*z1 + self.w2*z2 + self.b_g)
        return torch.clamp(g, self.g_min, self.g_max)

    def forward(self, x):  # x: [B,T,F_in]
        B, T, F = x.shape
        h = torch.zeros(B, self.d, device=x.device)
        g_val = self.gate()

        if F >= self.cal_dims:
            cal = x[..., -self.cal_dims:]         # [B,T,8]
            cal_aggr = cal.mean(dim=1)            # [B,8]
        else:
            cal_aggr = torch.zeros(B, self.cal_dims, device=x.device)
        cal_scalar = cal_aggr.mean(dim=1, keepdim=True)  # [B,1]

        for t in range(T):
            xt   = x[:, t, :]                    # [B,F_in]
            base = self.W(self.P(xt)) + self.b   # [B,d]
            base = base + self.alpha * cal_scalar
            base = self.ln(base)
            h    = (1 - g_val) * h + g_val * base
        return h

class QTrafficForecaster(nn.Module):
    def __init__(self, n_features_in, n_sensors, k=128, d=64, pred_len=720, drop_prob=0.2):
        super().__init__()
        self.backbone = QTemporal(n_features_in, k=k, d=d, cal_dims=CAL_DIMS, g_min=g_min, g_max=g_max)
        self.dec = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(d, pred_len * n_sensors)     # ← DOAR SENZORI la ieșire
        )
        self.pred_len = pred_len
        self.n_sensors = n_sensors
        kaiming_normal_all_linear(self)

    def forward(self, x):  # x: [B,T,F_in=sensors+8]
        h = self.backbone(x)                                  # [B,d]
        y_base = self.dec(h).view(x.size(0), self.pred_len, self.n_sensors)  # [B,H,F_sensors]
        last_obs = x[:, -1, :self.n_sensors].unsqueeze(1)     # [B,1,F_sensors] ← numai senzori
        return y_base + last_obs

model = QTrafficForecaster(n_features_in=F_in, n_sensors=F_sensors, k=k, d=d, pred_len=pred_len, drop_prob=drop_prob).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=4, factor=0.5)
lossf = nn.MSELoss()

print("Params:", sum(p.numel() for p in model.parameters()))

# -------------------- Train + Early Stopping --------------------
best_val = float('inf'); best_state = None; no_improve = 0

def evaluate_mse(dloader):
    s, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for xb, yb in dloader:
            xb = xb.to(device).float(); yb = yb.to(device).float()  # yb: [B,H,F_sensors]
            pred = model(xb)
            s += lossf(pred, yb).item() * xb.size(0); n += xb.size(0)
    return s / max(1, n)

for ep in range(1, epochs+1):
    model.train()
    tr_sum, n_tr = 0.0, 0
    for xb, yb in dl_train:
        xb = xb.to(device).float()
        yb = yb.to(device).float()     # [B,H,F_sensors]
        opt.zero_grad()
        pred = model(xb)
        loss = lossf(pred, yb)         # doar senzori
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        tr_sum += loss.item() * xb.size(0); n_tr += xb.size(0)
    tr_mse = tr_sum / max(1, n_tr)

    va_mse = evaluate_mse(dl_val)
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

# -------------------- Test (MAE/MSE pe senzori, NORMALIZED) --------------------
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in dl_test:
        xb = xb.to(device).float()
        pred = model(xb).cpu().numpy()     # [B,H,F_sensors]
        preds.append(pred); trues.append(yb.numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

MAE_all = mean_absolute_error(trues.reshape(-1, F_sensors),
                              preds.reshape(-1, F_sensors))
MSE_all = mean_squared_error(trues.reshape(-1, F_sensors),
                             preds.reshape(-1, F_sensors))

print("\n───────── TEST METRICS (Normalized, sensors only) ─────────")
print(f"MAE : {MAE_all:.6f}")
print(f"MSE : {MSE_all:.6f}")
