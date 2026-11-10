# Problem Setup & Information-Gain Guarantees for Energy–Force GP Active Learning

## 1. System Representation and Induced Kernel

Let a molecular configuration be $R \in \mathbb{R}^{3N_{\text{at}}}$.
We define the total potential energy surface $E(R)$ as a sum over local atomic contributions:
$$E(R) = \sum_{p=1}^{N_{\text{at}}} \epsilon(D_p(R))$$
where $D_p(R) \in \mathbb{R}^d$ is the smooth, symmetry-invariant Smooth Overlap of Atomic Positions (SOAP) descriptor for atom $p$, and $\epsilon \sim \mathcal{GP}(0, k)$ is a Gaussian Process over descriptor space with an RBF covariance kernel:
$$k(D, D') = \sigma_v^2 \exp\left( -\frac{\|D - D'\|^2}{2\ell^2} \right)$$

The induced configuration-space kernel $K: \mathbb{R}^{3N_{\text{at}}} \times \mathbb{R}^{3N_{\text{at}}} \to \mathbb{R}$ is:
$$K(R, R') = \sum_{p=1}^{N_{\text{at}}} \sum_{q=1}^{N_{\text{at}}} k(D_p(R), D_q(R'))$$

Because SOAP representations are continuous and twice differentiable with respect to atomic positions $R$, $K(R,R')$ is a valid, positive-definite, twice-differentiable kernel on configuration space.

## 2. Joint Energy–Force Observation Model

Atomic forces are negative gradients of total energy: $F(R) = -\nabla_R E(R)$.
For a query configuration $R_i$, the joint observation vector $y_i \in \mathbb{R}^{1 + 3N_{\text{at}}}$ is:
$$y_i = \begin{bmatrix} E(R_i) \\ F(R_i) \end{bmatrix} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \mathbf{\Omega})$$
where $\mathbf{\Omega} = \text{diag}(\sigma_E^2, \sigma_F^2 \mathbf{I}_{3N_{\text{at}}})$.

The block covariance between configuration $R_a$ and $R_b$ follows from linear differential operators acting on $K(R_a, R_b)$:
$$\text{Cov}(E^a, E^b) = K(R_a, R_b)$$
$$\text{Cov}(E^a, F^b_{m,k}) = -\frac{\partial K(R_a, R_b)}{\partial R^b_{m,k}}$$
$$\text{Cov}(F^a_{m,k}, F^b_{n,l}) = \frac{\partial^2 K(R_a, R_b)}{\partial R^a_{m,k} \partial R^b_{n,l}}$$

## 3. Information Gain and Covariance Decay Bounds

Following Yu et al. (2024), the maximum information gain $\gamma_t$ after $t$ joint energy-force queries is bounded by the effective dimension of the induced kernel matrix:
$$\gamma_t = \frac{1}{2} \log \det \left( \mathbf{I} + \mathbf{\Omega}^{-1} \mathbf{K}_t \right)$$

Because force observations inject $3N_{\text{at}}$ derivative measurements per configuration query, the predictive variance decay rate satisfies:
$$\sigma_t^2(R) = \mathcal{O}\left( \frac{\gamma_t}{t \cdot (1 + 3N_{\text{at}})} \right)$$

Under the SOAP mapping $D(R)$, rotational and translational symmetries restrict the active configuration space to an intrinsic manifold of dimension $d_{\text{eff}} \ll 3N_{\text{at}} - 6$, accelerating the eigenvalue decay rate of $\mathbf{K}_t$.
