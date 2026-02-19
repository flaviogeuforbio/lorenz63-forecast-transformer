from data.simulate import simulate_traj, plot_traj
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#function to sample windows from a simulated trajectory
def get_windows_from_traj(
        traj: np.array, 
        T: int, n_windows: int, 
        rng: np.random.Generator, 
        stop_idx: int = 2000 #to avoid dead windows in under-critical trajectories
):
    #check dimensions
    L = len(traj)
    if L < T:
        raise ValueError(f"[ERROR] Expected trajectory size ({L}) > windows size ({T})")

    #generate random window start idxs
    starts = rng.integers(0, stop_idx - T, size = n_windows) #windows will have some overlap
    windows = np.stack([traj[s:s+T] for s in starts], axis = 0) #extracting windows and converting to desidered dimensions [N, T, 3] (N = n_windows_for_traj here)

    return windows

#function to sample windows AND forecasting targets from a simulated trajectory
def get_forecast_pairs_from_traj(
    traj: np.array,
    T: int,
    H: int, 
    n_windows: int, 
    rng: np.random.Generator,
    stop_idx: int = 2000 
):
    
    #check dimensions
    total_len = T + H
    L = len(traj)
    if L < total_len:
        raise ValueError(f"[ERROR] Expected trajectory size ({L}) > windows size ({total_len})")
    
    #generate random window start idxs
    starts = rng.integers(0, stop_idx - total_len, size = n_windows)
    X = np.stack([traj[s:s + T] for s in starts], axis = 0)
    Y = np.stack([traj[s+T:s+T+H] for s in starts], axis = 0)

    return X, Y


#function to create train, val, test datasets
def create_dataset(
        rhos: np.array, #array with linearly spaced values of rho 
        n_traj_for_rho: int,
        n_windows_for_traj: int, #---windows sampling parameters
        T: int = 128, #---
        seed: int = 0, #for Generator object --> this will differentiate train, test, val dataset
        sigma: float = 10.0, #---simulation parameters
        beta: float = 8.0/3.0,
        rho_c: float = 24.7,
        dt: float = 0.01,
        n_steps: int = 25000, 
        burn_in: int = 400, #---
        #the burn_in value is not randomly chosen but its choice is driven by the analysis conducted in analysis.py 
        #(two main effects: 1. avoid dead windows in under-critical, 2. capture upper-critical transient as good and informative physical data)
):
    
    rng = np.random.default_rng(seed) #creating the Generator -> used for initial conditions & windows sampling

    all_windows, rho_values, regime_labels = [], [], [] #initializing arrays
    
    #main loop
    for rho in tqdm(rhos):
        for _ in range(n_traj_for_rho):
            #generating trajectory
            x0 = rng.integers(low = -5.0, high = 5.0, size = 3) #initial conditions in a small 10x10x10 box centered in the origin of the axis

            traj = simulate_traj(rho=rho, sigma=sigma, beta=beta, dt=dt, n_steps=n_steps, burn_in=burn_in, x0=x0)
            traj_windows = get_windows_from_traj(traj=traj, T=T, n_windows=n_windows_for_traj, rng=rng) #sampling windows from the trajectory

            #updating arrays of data (windows) & desired output/labels
            all_windows.append(traj_windows)
            rho_values.append(np.full((n_windows_for_traj,), rho, dtype = np.float32))
            regime_labels.append(np.full((n_windows_for_traj, ), 1.0 if rho > rho_c else 0.0, dtype = np.float32))
    
    #stacking arrays to match desired dimensions
    all_windows = np.concatenate(all_windows, axis = 0, dtype = np.float32) # [N, T, 3], where N = len(rhos) * n_traj_for_rho * n_windows_for_traj (total n. of windows)
    rho_values = np.concatenate(rho_values, axis = 0, dtype = np.float32) # [N, 1]
    regime_labels = np.concatenate(regime_labels, axis = 0, dtype = np.float32)# [N, 1]

    return all_windows, rho_values, regime_labels


def create_forecast_dataset(
    rhos: np.array, 
    n_traj_for_rho: int, 
    n_windows_for_traj: int, 
    T: int = 128,
    H: int = 32, 
    seed: int = 0,
    sigma: float = 10.0, 
    beta: float = 8.0/3.0, 
    dt: float = 0.01, 
    n_steps: int = 25000, 
    burn_in: int = 400
):
    
    rng = np.random.default_rng(seed)

    all_X, all_Y = [], []

    #main loop 
    for rho in tqdm(rhos):
        for _ in range(n_traj_for_rho):
            x0 = rng.integers(low = -5.0, high = 5.0, size = 3)

            traj = simulate_traj(rho=rho, sigma=sigma, beta=beta, dt=dt, n_steps=n_steps, burn_in=burn_in, x0=x0)
            traj_X, traj_Y = get_forecast_pairs_from_traj(traj=traj, T=T, H=H, n_windows=n_windows_for_traj, rng=rng)

            all_X.append(traj_X)
            all_Y.append(traj_Y)

    all_X = np.concatenate(all_X, axis = 0, dtype = np.float32) #[N, T, 3]
    all_Y = np.concatenate(all_Y, axis = 0, dtype = np.float32)

    return all_X, all_Y
    


#function to return standardized datasets (using train statistics to avoid data leakage)
def apply_train_standardization(
        X_train: np.array, #--- datasets
        X_val: np.array,
        X_test: np.array, #---
        eps: float = 1e-6 #to avoid zero-division issue
    ):

    #calculating statistics on train dataset
    mean = X_train.reshape(-1, 3).mean(axis = 0) #(3, )
    std = X_train.reshape(-1, 3).std(axis = 0) + eps #(3, )

    #applying standardization to all the datasets
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test


def apply_train_standardization_forecast(
    X_train: np.array, X_val: np.array, X_test: np.array, 
    Y_train: np.array, Y_val: np.array, Y_test: np.array,
    eps: float = 1e-6 
):
    
    mean = X_train.reshape(-1, 3).mean(axis = 0) #reshape: [N, T, 3] -> [N*T, 3]
    std = X_train.reshape(-1, 3).std(axis = 0) + eps

    def norm(A): return (A - mean) / std

    return norm(X_train), norm(Y_train), norm(X_val), norm(Y_val), norm(X_test), norm(Y_test), mean.astype(np.float32), std.astype(np.float32)
    

if __name__ == "__main__":
    #array of linearly spaced values of rho (50 values in range [5, 40] -> symmetrical around rho_c = 24.7)
    rhos = np.linspace(5, 40, 50)

    #creating train, val, test data for regression/classification
    X_train, rho_train, regime_train = create_dataset(rhos=rhos, n_traj_for_rho=6, n_windows_for_traj=40, seed=0)
    X_val, rho_val, regime_val = create_dataset(rhos=rhos, n_traj_for_rho=1, n_windows_for_traj=40, seed=1)
    X_test, rho_test, regime_test = create_dataset(rhos=rhos, n_traj_for_rho=1, n_windows_for_traj=40, seed=2)

    #creating train, val, test data for forecasting task
    Xf_train, Yf_train = create_forecast_dataset(rhos=rhos, n_traj_for_rho=6, n_windows_for_traj=40, seed = 0)
    Xf_val, Yf_val = create_forecast_dataset(rhos=rhos, n_traj_for_rho=1, n_windows_for_traj=40, seed = 1)
    Xf_test, Yf_test = create_forecast_dataset(rhos=rhos, n_traj_for_rho=1, n_windows_for_traj=40, seed = 2)

    #standardizing data
    X_train, X_val, X_test = apply_train_standardization(X_train, X_val, X_test)
    Xf_train, Yf_train, Xf_val, Yf_val, Xf_test, Yf_test, mean, std = apply_train_standardization_forecast(Xf_train, Xf_val, Xf_test, Yf_train, Yf_val, Yf_test)

    #saving dataset
    np.savez_compressed(
        "lorenz_dataset.npz",
        X_train=X_train, rho_train=rho_train, regime_train=regime_train, #regr/classification task
        X_val=X_val, rho_val=rho_val, regime_val=regime_val,             #..
        X_test=X_test, rho_test=rho_test, regime_test=regime_test,       #..
        Xf_train=Xf_train, Yf_train=Yf_train, Xf_val=Xf_val, Yf_val=Yf_val, Xf_test=Xf_test, Yf_test=Yf_test, #forecasting task
        mean = mean, std = std                                                                                #..   
    )
