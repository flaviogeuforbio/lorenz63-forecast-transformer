import numpy as np
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from data.simulate import simulate_traj
from data.build_dataset import get_forecast_pairs_from_traj

def destandardize(arr, mean, std):
    return arr * std + mean

def persistence(x, H):
    return x[:, -1:, :].repeat(1, H, 1)

def perh_loss(y_pred, y):
    return ((y_pred - y)**2).mean(dim = (0, 2))

def plot_training_curves(file_path, outpath):
    data = np.load(file_path)

    train = data["train"]
    val = data["val"]
    best_epoch = int(data["best_epoch"][0])

    epochs = np.arange(1, len(train) + 1)

    plt.figure(figsize = (6, 4), dpi = 640)

    plt.title("Training & validation loss")
    plt.plot(epochs, train, label = "Train", color = "green", alpha = 0.6)
    plt.plot(epochs, val, label = "Validation", color = "red", alpha = 0.6)
    plt.scatter([best_epoch + 1], [val[best_epoch]], label = "Best checkpoint", color = "red", zorder = 3, alpha = 0.9)
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log)")
    plt.yscale("log")

    plt.legend()
    plt.tight_layout()

    plt.savefig(outpath)
    print(f"Saved: {outpath}")
    plt.close()


@torch.no_grad()
def analyze_by_regime(model, val_dl, device, pers_comparison: bool = False, H: int = 32):
    model.eval()

    perh_loss_0 = torch.zeros(size = (H,))
    perh_loss_1 = torch.zeros(size = (H,))
    perh_loss_0_pers = torch.zeros(size = (H,))
    perh_loss_1_pers = torch.zeros(size = (H,))

    for x, y, rho, regime in tqdm(val_dl, leave = False):
        #moving tensors to device
        x = x.to(device)
        y = y.to(device)
        regime = regime.to(device)

        #creating masks
        mask0 = (regime < 0.5)
        mask1 = ~mask0

        #calculate predictions
        y_pred, _ = model(x)
        if pers_comparison:
            y_pers = persistence(x, H)

        #calculating perh losses
        if mask0.any():
            y_pred_0 = y_pred[mask0]
            perh_loss_0 += perh_loss(y_pred_0, y[mask0])
            if pers_comparison: perh_loss_0_pers += perh_loss(y_pers[mask0], y[mask0])

        if mask1.any():
            y_pred_1 = y_pred[mask1]
            perh_loss_1 += perh_loss(y_pred_1, y[mask1])
            if pers_comparison: perh_loss_1_pers += perh_loss(y_pers[mask1], y[mask1])
        

    if pers_comparison: return perh_loss_0, perh_loss_1, perh_loss_0_pers, perh_loss_1_pers
    return perh_loss_0, perh_loss_1


def plot_mse_per_step_by_regime(perh_loss_0, perh_loss_0_pers, perh_loss_1, perh_loss_1_pers, outpath: str, H: int = 32):
    t = np.arange(1, H + 1)
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))

    axs[0].set_title("Non-chaotic (0)")
    axs[0].set_xlabel("Forecast step")
    axs[0].set_ylabel("MSE")
    axs[0].plot(t, perh_loss_0, label = "Transformer")
    axs[0].plot(t, perh_loss_0_pers, label = "Baseline (persistence)")
    axs[0].legend()

    axs[1].set_title("Chaotic (1)")
    axs[1].set_xlabel("Forecast step")
    axs[1].set_ylabel("MSE (log)")
    axs[1].plot(t, perh_loss_1, label = "Transformer")
    axs[1].plot(t, perh_loss_1_pers, label = "Baseline (persistence)")
    axs[1].set_yscale("log")
    axs[1].legend()

    fig.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok = True)
    fig.savefig(outpath, bbox_inches = "tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

@torch.no_grad()
def rollout(model, x0, steps: int = 300, stride: int = 8):
    """
       Auto-regressive rollout to generate a forecast trajectory much longer than model output lenght (H = 32 by default).
    
       model: Transfomer model used to do forecasting\n
       x0: input trajectory window\n 
       steps: rollout total lenght\n
       stride: tokens taken per each prediction
    """
    model.eval()

    x_win = x0.clone() #[1, T, 3]
    preds = []

    produced = 0
    while produced < steps:
        y_pred, _ = model(x_win)
        take = min(stride, steps - produced)
        preds.append(y_pred[:, :take, :])

        produced += take
        x_win = torch.cat([x_win[:, take:, :], y_pred[:, :take, :]], dim = 1)

    preds = torch.cat(preds, dim = 1)
    return preds


@torch.no_grad()
def make_gif_traj(
    model, 
    mean, 
    std, 
    device, 
    forecast_lenght: int, 
    rollout_stride: int = 8,
    T: int = 128, 
    stop_idx: int = 2000,
    regime: int = 1, 
    seed: int = 0
): #by default we consider chaotic traj
    
    rng = np.random.default_rng(seed)
    
    #note: pho_c = 24.7
    if regime == 1:
        rho = rng.uniform(28.0, 35.0)
    elif regime == 0:
        rho = rng.uniform(10.0, 20.0)
    else:
        print(f"Regime: {regime} not valid, must be 0 or 1.")
        return

    traj = simulate_traj(rho)
    x_input, y_true = get_forecast_pairs_from_traj(traj=traj, T=T, H=forecast_lenght, n_windows = 1, rng = rng, stop_idx = stop_idx)

    x_input = torch.tensor(x_input, dtype=torch.float32)
    x_input = (x_input - mean) / std
    x_input = x_input.to(device)

    y_pred = rollout(model, x_input, steps = forecast_lenght, stride = rollout_stride)                # [1,f,3]
    # optionally compare with persistence:
    # y_pers = persistence(x, H)

    x_input = destandardize(x_input, mean, std)
    y_pred = destandardize(y_pred, mean, std)

    x_np = x_input.squeeze(0).detach().cpu().numpy()
    y_true_np = y_true.reshape((forecast_lenght, 3)) # [f,3]
    y_pred_np = y_pred.squeeze(0).detach().cpu().numpy()  # [f,3]

    return x_np, y_true_np, y_pred_np, traj



def make_gif(x: np.array, y_true: np.array, y_pred: np.array, traj: np.array, outpath: str, T: int = 128, regime: int = 1, context_lenght: int = 5000, burn_in: int = 500, max_batches=50): 

    # Axis ranges (fixed) for a stable, clean animation
    all_pts = np.vstack([x, y_true, y_pred])
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)
    pad = 0.05 * (all_pts.max(axis=0) - all_pts.min(axis=0) + 1e-9)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    title = "Chaotic (1)" if regime == 1 else "Non-chaotic (0)"
    ax.set_title(f"True vs Forecast — {title}")

    context = traj[burn_in:context_lenght, :]

    # plot history window as context
    ax.scatter(context[:,0], context[:,1], context[:,2], s=1, alpha=0.1, label="Context trajectory")
    ax.plot(x[:,0], x[:,1], x[:,2], linewidth=1.5, alpha=0.9, label="History (Input)")

    true_line, = ax.plot([], [], [], linewidth=2.0, label="True future")
    pred_line, = ax.plot([], [], [], linewidth=2.0, label="Forecast (Transformer)")

    info_txt = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.65, edgecolor="none"))

    ax.set_xlim(xmin - pad[0], xmax + pad[0])
    ax.set_ylim(ymin - pad[1], ymax + pad[1])
    ax.set_zlim(zmin - pad[2], zmax + pad[2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")

    ax.view_init(elev=20, azim=-60)

    #calculate metrics to show during animation
    mse_per_step = np.mean((y_true - y_pred)**2, axis = 1) #mean squared error per step
    rmse_cum = np.sqrt(np.cumsum(mse_per_step) / (np.arange(len(mse_per_step)) + 1)) #cumulative root mean squared error 
    L = len(mse_per_step)

    def init():
        true_line.set_data([], [])
        true_line.set_3d_properties([])
        pred_line.set_data([], [])
        pred_line.set_3d_properties([])
        info_txt.set_text("")
        return true_line, pred_line, info_txt

    def update(frame):
        k = frame + 1
        true_line.set_data(y_true[:k,0], y_true[:k,1])
        true_line.set_3d_properties(y_true[:k,2])

        pred_line.set_data(y_pred[:k,0], y_pred[:k,1])
        pred_line.set_3d_properties(y_pred[:k,2])

        info_txt.set_text(
            f"step: {k}/{L}\n"
            f"mse@step: {mse_per_step[k-1]:.4f}\n"
            f"rmse(cum): {rmse_cum[k-1]:.4f}"
        )

        return true_line, pred_line, info_txt

    anim = FuncAnimation(fig, update, frames=len(y_pred), init_func=init, interval=80, blit=False)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    anim.save(outpath, writer=PillowWriter(fps=12))
    plt.close(fig)
    print("Saved:", outpath)