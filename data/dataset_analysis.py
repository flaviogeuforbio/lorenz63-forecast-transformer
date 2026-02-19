from utils import check_dead_windows_fraction, check_optimal_windows_start, compare_regimes_meanstd
from dataset import make_loaders

#importing dataloaders (train, val, test)
train_dl, val_dl, test_dl = make_loaders(num_workers = 0)

#checking dead windows fraction in datasets (to tune stop_idx parameter for windows sampling)
print(f"Train dataset dead windows fraction: {check_dead_windows_fraction(train_dl):.3f}")
print(f"Validation dataset dead windows fraction: {check_dead_windows_fraction(val_dl):.3f}")
print(f"Test dataset dead windows fraction: {check_dead_windows_fraction(test_dl):.3f}")

#the idea is to graphically evaluate a good value for burn_in parameter (cutting the first transient)
check_optimal_windows_start(outpath = "plots/optimal_windows_start.png")

#mean std is noticeably different between sub-critical and super-critical windows --> regime classification is an easy task
compare_regimes_meanstd(outpath = "plots/meanstd_by_regime.png", dataloader = val_dl, dataset_name = "Validation")

