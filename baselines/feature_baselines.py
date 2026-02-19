import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy() #detach from computational graph and convert to numpy

    return x

def collect_dataset(dataloader: DataLoader):
    all_windows = []
    all_rho = []
    all_regimes = []

    for windows, rho, regimes in dataloader:
        #move to cpu and convert to numpy
        windows = _to_numpy(windows)
        rho = _to_numpy(rho)
        regimes = _to_numpy(regimes)

        #update lists
        all_windows.append(windows)
        all_rho.append(rho)
        all_regimes.append(regimes)

    #concatenate lists elements into np.ndarray
    all_windows = np.concatenate(all_windows, axis = 0)
    all_rho = np.concatenate(all_rho, axis = 0)
    all_regimes = np.concatenate(all_regimes, axis = 0)

    return all_windows, all_rho, all_regimes


def extract_features(X: np.ndarray) -> np.ndarray:
    """Takes as input an array of temporal windows [N, T, 3] and returns an array of features [N, F]"""

    #basic features (static)
    mean = X.mean(axis = 1) #[N, 3]
    std = X.std(axis = 1) #[N, 3]
    xmin = X.min(axis = 1) #[N, 3]
    xmax = X.max(axis = 1) #[N, 3]

    #dynamical features
    dX = np.diff(X, axis = 1) #[N, T-1, 3] (unitary speed, dT = 1)
    mabs_dX = np.mean(np.abs(dX), axis = 1) #[N, 3]
    std_dX = np.std(dX, axis = 1) #[N, 3]

    #cross-channel covariance lag 0 (same time)
    def cross_cov(a, b, epsilon: float = 1e-8):
        """Expects 2 2d array of size [N,T] -> returns their covariance with size [N, 1]"""
        a0 = a - a.mean(axis = 1, keepdims = True)
        b0 = b - b.mean(axis = 1, keepdims = True)
        cov = np.mean(a0 * b0, axis = 1) #[N,]
        denom = (a.std(axis = 1) * b.std(axis = 1) + epsilon)
        return (cov / denom)[:, None] #[N, 1]
    
    #extract x(t), y(t), z(t) for each window in dataset
    x = X[:, :, 0]
    y = X[:, :, 1]
    z = X[:, :, 2]

    #collect all covariances
    covxy = cross_cov(x, y)
    covxz = cross_cov(x, z)
    covyz = cross_cov(y, z)

    #putting all features together
    feats = np.concatenate([mean, std, xmin, xmax, mabs_dX, std_dX, covxy, covxz, covyz], axis = 1)
    return feats #[N, F]


def run_feature_baselines(train_loader: DataLoader, val_loader: DataLoader):
    #collecting datasets
    X_train, rho_train, reg_train = collect_dataset(train_loader)
    X_val, rho_val, reg_val = collect_dataset(val_loader)

    #extracting features to fit baseline models
    F_train = extract_features(X_train)
    F_val = extract_features(X_val)

    # --- RIDGE REGRESSION (for rho continuos value)
    regr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha = 1.0)) #using standard hyperparameter
    ])

    #predicting validation set rho
    regr_model.fit(F_train, rho_train)
    rho_pred = regr_model.predict(F_val)

    #calculating metrics
    mae = mean_absolute_error(rho_val, rho_pred)
    rmse = mean_squared_error(rho_val, rho_pred) ** 0.5
    r2 = r2_score(rho_val, rho_pred)

    # --- LOGISTIC REGRESSION (for regime classification)
    clf_model = Pipeline([
        ("scaler", StandardScaler()), 
         ("logistic", LogisticRegression(max_iter = 2000))
    ])

    #predicting validation set regimes
    clf_model.fit(F_train, reg_train)
    reg_pred = clf_model.predict(F_val)

    #calculating metrics
    accuracy = accuracy_score(reg_val, reg_pred)
    f1 = f1_score(reg_val, reg_pred)

    print("=== Feature-only baselines (VAL) ===")
    print(f"[rho] MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}")
    print(f"[reg] ACC={accuracy:.3f} | F1={f1:.3f}")

    return regr_model, clf_model


def ridge_error_analysis(regr_model: Pipeline, val_loader: DataLoader, size: int = 5):
    #isolate rho bins for prediction
    bins = [[i, i + size] for i in np.arange(5, 40, size)]

    maes = [] #mean absolute errors
    centers = [] #bins center values
    meanstds = [] #bins windows mean std

    #getting data & extracting features
    X, rho, _ = collect_dataset(val_loader)
    F = extract_features(X)

    for i, (low, high) in enumerate(bins):
        #creating mask & selecting desidered samples
        mask = (rho >= low) & (rho < high) if i < (len(bins) - 1) else (rho >= low) & (rho <= high)
        F_masked = F[mask]
        rho_masked = rho[mask]

        #calculating mean std for each bin
        F_masked_std = F_masked[:, 3:6] #isolating std features
        meanstd = F_masked_std.mean() #mean over both axis
        meanstds.append(meanstd)

        #predicting rho & calculating mae
        rho_pred = regr_model.predict(F_masked)
        bin_mae = mean_absolute_error(rho_masked, rho_pred)

        #saving data
        maes.append(bin_mae)
        centers.append((low + high) / 2)

    #plot and return results    
    fig, ax1 = plt.subplots(figsize = (6, 4))
    # Primo asse Y: MAE
    ax1.bar(
        centers,
        maes,
        width=size,
        alpha=0.6,
        color="tab:blue",
        edgecolor="black",
        label="MAE"
    )
    ax1.set_xlabel(r"$\rho$")
    ax1.set_ylabel("MAE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Secondo asse Y: mean(std)
    ax2 = ax1.twinx()
    ax2.bar(
        centers,
        meanstds,
        width = size * 0.6,
        alpha=0.6,
        color="tab:orange",
        edgecolor="black",
        label="Mean $ \sigma $"
    )
    ax2.set_ylabel("Mean window std", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("MAE and mean window std vs $\\rho$")
    plt.tight_layout()
    plt.show()

    return centers, maes, meanstds
    