"""
tests/sanity_check_derivative_gp.py

Strengthened sanity check: fit a GPyTorch derivative-GP to a
multi-frequency (non-trivially-extrapolable) 1D function, with a gap
much wider than the function's fastest oscillation. Unlike a plain
sin(x), the GP cannot "cheat" by matching slopes at the gap edges -
a short lengthscale is needed to fit the fast components, which
means genuine uncertainty growth is expected deep inside the gap.

Run from the project root:
    python tests/sanity_check_derivative_gp.py
"""

import torch
import gpytorch
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(0)


# ---------------------------------------------------------------
# 1. Ground truth: sum of three incommensurate frequencies.
#    Periods are ~6.28, ~1.90, ~1.10 - the gap below (width 5) is
#    several periods of the fastest component, so confidently
#    extrapolating across it from the edges alone is not possible.
# ---------------------------------------------------------------
FREQS = (1.0, 3.3, 5.7)
AMPS = (1.0, 0.3, 0.15)


def true_function(x):
    return sum(a * torch.sin(f * x) for f, a in zip(FREQS, AMPS))


def true_derivative(x):
    return sum(a * f * torch.cos(f * x) for f, a in zip(FREQS, AMPS))


# Dense sampling on the left, a wide unsampled gap, sparse sampling
# on the right.
train_x = torch.cat([
    torch.linspace(0.0, 3.0, 10),
    torch.linspace(8.0, 12.0, 5),
])
train_y_val = true_function(train_x)
train_y_grad = true_derivative(train_x)

noise_std = 0.01
train_y_val = train_y_val + noise_std * torch.randn_like(train_y_val)
train_y_grad = train_y_grad + noise_std * torch.randn_like(train_y_grad)

train_y = torch.stack([train_y_val, train_y_grad], dim=-1)


# ---------------------------------------------------------------
# 2. Model (unchanged from the first sanity check)
# ---------------------------------------------------------------
class DerivativeGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = DerivativeGPModel(train_x, train_y, likelihood)


# ---------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iter = 300
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    if (i + 1) % 50 == 0:
        print(f"iter {i + 1:3d}/{n_iter} - loss: {loss.item():.4f}")

learned_lengthscale = model.covar_module.base_kernel.lengthscale.item()
print(f"\nLearned RBF lengthscale: {learned_lengthscale:.4f}")
print("(fastest true period is ~1.10 - compare against this)")


# ---------------------------------------------------------------
# 4. Predict across the full domain, including the gap
# ---------------------------------------------------------------
model.eval()
likelihood.eval()

test_x = torch.linspace(0.0, 12.0, 400)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(test_x))
    mean = preds.mean
    lower, upper = preds.confidence_region()

pred_f_mean, pred_df_mean = mean[:, 0], mean[:, 1]
pred_f_lower, pred_df_lower = lower[:, 0], lower[:, 1]
pred_f_upper, pred_df_upper = upper[:, 0], upper[:, 1]


# ---------------------------------------------------------------
# 5. Diagnostics
# ---------------------------------------------------------------
true_f = true_function(test_x)
true_df = true_derivative(test_x)
var = preds.variance

dense_region = (test_x > 1.0) & (test_x < 2.0)        # inside the densely-sampled zone
gap_center = (test_x > 5.0) & (test_x < 6.0)            # deep inside the wide gap

var_dense = var[dense_region, 0].mean().item()
var_gap = var[gap_center, 0].mean().item()
ratio = var_gap / var_dense

rmse_dense = torch.sqrt(torch.mean((pred_f_mean[dense_region] - true_f[dense_region]) ** 2)).item()
rmse_gap = torch.sqrt(torch.mean((pred_f_mean[gap_center] - true_f[gap_center]) ** 2)).item()

print("\n--- Sanity check results ---")
print(f"RMSE in densely-sampled region:  {rmse_dense:.4f}")
print(f"RMSE deep in the gap:            {rmse_gap:.4f}  (expected to be larger - this is the point)")
print(f"Predictive variance, dense region: {var_dense:.5f}")
print(f"Predictive variance, gap center:   {var_gap:.5f}")
print(f"Variance ratio (gap / dense):       {ratio:.2f}x")

assert rmse_dense < 0.1, "Fit in the sampled region is worse than expected - check training"
assert var_gap > var_dense, "Predictive variance should be larger in the unsampled gap"

if ratio < 3.0:
    print(
        "\nNOTE: variance ratio is positive but still fairly weak "
        "(<3x). Inspect the plot - if the shaded band in the gap "
        "still looks nearly flat, the function may still be too "
        "easy to extrapolate; worth discussing before moving on."
    )
else:
    print(f"\nStrong separation ({ratio:.1f}x) - the model is honestly "
          f"uncertain in the gap. Sanity check passed convincingly.")


# ---------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(test_x, true_f, "k--", label="true f(x)")
axes[0].plot(test_x, pred_f_mean, "b", label="GP mean")
axes[0].fill_between(test_x, pred_f_lower, pred_f_upper, alpha=0.2, color="b")
axes[0].scatter(train_x, train_y_val, c="r", marker="x", label="observed", zorder=5)
axes[0].axvspan(3.0, 8.0, color="gray", alpha=0.1, label="unsampled gap")
axes[0].set_title("Function value")
axes[0].legend(fontsize=8)

axes[1].plot(test_x, true_df, "k--", label="true f'(x)")
axes[1].plot(test_x, pred_df_mean, "b", label="GP mean")
axes[1].fill_between(test_x, pred_df_lower, pred_df_upper, alpha=0.2, color="b")
axes[1].scatter(train_x, train_y_grad, c="r", marker="x", label="observed", zorder=5)
axes[1].axvspan(3.0, 8.0, color="gray", alpha=0.1, label="unsampled gap")
axes[1].set_title("Derivative")
axes[1].legend(fontsize=8)

plt.tight_layout()

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "sanity_check_derivative_gp.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path.resolve()}")