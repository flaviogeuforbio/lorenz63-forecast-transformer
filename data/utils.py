from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os

from simulate import simulate_traj

#function to check how many windows represent a trivial (static) trajectory -> to tune 'stop_idx' hyperparameter (or the windows sampling algorithm)
@torch.no_grad()
def check_dead_windows_fraction(dataloader: DataLoader, rho_c: float = 24.7, epsilon: float = 1e-2):
    dead = 0
    subcritical = 0

    for x, rho, _ in dataloader:
        # x: (B,T,3), rho: (B,)
        mask = rho < rho_c
        if mask.any():
            x_sub = x[mask]  # (B_sub, T, 3)
            # std over time dimension -> (B_sub, 3)
            std = x_sub.std(dim=1)  # dim=1 is time
            is_dead = (std < epsilon).all(dim=1)  # (B_sub,)
            dead += int(is_dead.sum().item())
            subcritical += int(mask.sum().item())

    return dead / subcritical if subcritical > 0 else 0.0

#auxiliary
def windows_std_by_startidx(traj: torch.tensor, T: int = 128, stop_idx: int = 5000):
    start_idxs = torch.arange(0, stop_idx + 1)

    std_values = []

    for idx in start_idxs:
        window = traj[idx: idx + T]
        std = window.std(dim = 0).mean().item()
        std_values.append(std)

    return start_idxs.tolist(), std_values

def check_optimal_windows_start(
    outpath: str,
    rhos: list = [5.0, 12.0, 21.0], #optimal burn_in & start_idx range depend on rho value (and mostly on regime)
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    n_steps: int = 25000,
    burn_in: int = 0 #the idea is we want to tune the burn_in so we set it to zero now
):
    #creating trajectory with rhos < rho_c = 24.7
    for rho in rhos:
        traj = simulate_traj(rho, sigma, beta, dt, n_steps, burn_in)
        traj = torch.tensor(traj)

        starting_idxs, mean_wind_stds = windows_std_by_startidx(traj)

        plt.plot(starting_idxs, mean_wind_stds, label = f"rho={rho}")

    plt.xlabel("Starting idx")
    plt.ylabel("Mean window std.")

    plt.legend()
    
    os.makedirs(os.path.dirname(outpath), exist_ok = True)
    plt.savefig(outpath)
    plt.close()


def compare_regimes_meanstd(dataloader: DataLoader, dataset_name: str, outpath: str, rho_c: float = 24.7):
    mean_std_sub = []
    mean_std_sup = []

    for windows, rho, _ in dataloader:
        mask = rho < rho_c #individuates sub-critical trajectories
        if mask.any():
            wind_sub = windows[mask]

            #populating subcritical windows std values array
            batch_mean_std_sub = wind_sub.std(dim = 1).mean(dim = 1) #(B, )
            mean_std_sub.append(batch_mean_std_sub)

        #populating super-critical windows std values array
        inv_mask = ~mask 
        if inv_mask.any():
            wind_sup = windows[inv_mask]
            batch_mean_std_sup = wind_sup.std(dim = 1).mean(dim = 1)
            mean_std_sup.append(batch_mean_std_sup)

    #concatenate torch tensor and convert them to lists
    mean_std_sub = torch.cat(mean_std_sub, dim = 0).tolist()
    mean_std_sup = torch.cat(mean_std_sup, dim = 0).tolist()

    plt.title(f"Mean std in sub-critical vs super-critical windows ({dataset_name})")

    plt.hist(mean_std_sub, color = "red", label = "sub-critical")
    plt.hist(mean_std_sup, color = "green", label = "super-critical")
    
    plt.ylabel("Occurrencies")
    plt.xlabel("Mean windows $ \sigma $")

    plt.legend()

    os.makedirs(os.path.dirname(outpath), exist_ok = True)
    plt.savefig(outpath)
    plt.close()


