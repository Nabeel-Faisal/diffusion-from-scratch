# Denoising Diffusion Probabilistic Models: Derivation and Implementation Notes

This document derives the mathematical foundations of the model implemented in this repository: an unconditional DDPM (Phases 1–2) extended to a text-conditional model via CLIP embeddings, cross-attention, and classifier-free guidance (Phase 3). Notation follows Ho et al. (2020) and Nichol & Dhariwal (2021) where applicable, with pointers to the corresponding code.

## 1. The Forward Process

Let $x_0 \sim q(x_0)$ be a sample from the data distribution (a $32 \times 32$ or $28 \times 28$ image, depending on phase). The forward process is a fixed Markov chain that gradually adds Gaussian noise over $T$ timesteps according to a variance schedule $\beta_1, \dots, \beta_T \in (0, 1)$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\ \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I\big) \tag{1}$$

Define $\alpha_t := 1 - \beta_t$ and $\bar\alpha_t := \prod_{s=1}^t \alpha_s$. Because the composition of Gaussians is Gaussian, the marginal $q(x_t \mid x_0)$ has a closed form that does not require simulating the chain step by step:

$$q(x_t \mid x_0) = \mathcal{N}\big(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I\big) \tag{2}$$

Equivalently, using the reparameterization trick with $\epsilon \sim \mathcal{N}(0, I)$:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon \tag{3}$$

Equation (3) is the basis of `q_sample` in `src/utils/diffusion.py`: given a clean image and a timestep, a noisy version is produced in a single operation, which is what makes training tractable — there is no need to step through $t$ intermediate states.

As $t \to T$, $\bar\alpha_t \to 0$ and $x_t$ approaches pure noise $\mathcal{N}(0, I)$, provided the schedule is designed so that $\bar\alpha_T$ is sufficiently small. This is the condition that makes the reverse process well-defined: at $t=T$ we can sample from a tractable prior and propagate backward.

### 1.1 Choice of Schedule

Two schedules are implemented in `NoiseScheduler`:

**Linear** (Ho et al. 2020): $\beta_t$ increases linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.

**Cosine** (Nichol & Dhariwal 2021): $\bar\alpha_t$ is defined directly as a function of $t$, and $\beta_t$ is derived from it:

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right) \tag{4}$$

$$\beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}, \quad \text{clipped to } \beta_t \le 0.999 \tag{5}$$

where $s$ is a small offset (0.008) preventing $\beta_t$ from being too small near $t=0$. The practical motivation: under the linear schedule, $\bar\alpha_{T/2} \approx 0.08$ in this implementation, meaning the signal is already mostly destroyed by the midpoint of the chain — many timesteps are spent on near-pure-noise inputs that contribute little new information to the model. The cosine schedule retains $\bar\alpha_{T/2} \approx 0.49$, distributing the destruction of signal more evenly across $t$, which empirically improves sample quality, particularly at low resolutions. This is consistent with what was observed when comparing Phase 1 (linear) and Phase 2 (cosine) training runs in this repository.

## 2. The Reverse Process

Since $q(x_{t-1} \mid x_t)$ is intractable (it requires integrating over the entire data distribution), it is approximated by a learned Gaussian:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\big) \tag{6}$$

Generation proceeds by sampling $x_T \sim \mathcal{N}(0, I)$ and applying $p_\theta$ iteratively down to $x_0$.

### 2.1 Variational Lower Bound

Training $p_\theta$ to match the true (but intractable) reverse process is framed as maximizing a variational lower bound on $\log p_\theta(x_0)$, analogous to a VAE with a fixed, non-learned encoder ($q$). The bound decomposes into a sum of KL divergences between $q(x_{t-1} \mid x_t, x_0)$ — the *true* posterior, which **is** tractable because it conditions on $x_0$ — and $p_\theta(x_{t-1} \mid x_t)$:

$$L = \mathbb{E}_q\left[\underbrace{D_{KL}(q(x_T \mid x_0) \,\|\, p(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{KL}(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}} \underbrace{- \log p_\theta(x_0 \mid x_1)}_{L_0}\right] \tag{7}$$

$L_T$ has no trainable parameters (both sides are close to the standard normal by construction of the schedule) and is dropped. The terms $L_{t-1}$ each compare two Gaussians, for which the KL divergence has a closed form depending only on the means (the variances are fixed to $\sigma_t^2 = \beta_t$ in the basic formulation, sidestepping the need to learn $\Sigma_\theta$). The posterior mean of $q(x_{t-1} \mid x_t, x_0)$ is:

$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t} x_0 \tag{8}$$

### 2.2 Reparameterizing in Terms of Noise

Substituting $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\right)$ (the inverse of Eq. 3) into Eq. 8 and simplifying yields a posterior mean expressed purely in terms of $x_t$ and the noise $\epsilon$ that produced it:

$$\tilde\mu_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right) \tag{9}$$

This motivates parameterizing the model not as a direct predictor of $\mu_\theta$, but as a noise predictor $\epsilon_\theta(x_t, t)$, with the model's mean prediction defined by substituting $\epsilon_\theta$ for $\epsilon$ in Eq. 9:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) \tag{10}$$

This is exactly the computation implemented in `SamplingCoeffs` and the DDPM sampling step in `src/utils/sampling.py`: `coeff_x = 1/sqrt(alpha_t)` and `coeff_eps = beta_t / sqrt(1 - alpha_bar_t)` correspond directly to the two coefficients in Eq. 10.

### 2.3 The Simplified Training Objective

With this parameterization, Ho et al. show that minimizing the KL terms $L_{t-1}$ reduces (up to a $t$-dependent weighting that is dropped for simplicity, and found empirically to work better unweighted) to:

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t \sim U\{1,T\},\, x_0,\, \epsilon \sim \mathcal{N}(0,I)}\left[\left\| \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big) \right\|^2\right] \tag{11}$$

This is the entire training procedure: sample a timestep and noise, construct $x_t$ via Eq. 3, ask the network to predict the noise that was added, and minimize mean-squared error. There is no need to evaluate Eq. 7 directly — Eq. 11 is what `scripts/train.py` implements as the training loop. The network $\epsilon_\theta$ is the U-Net in `src/models/unet.py`; conditioning on $t$ is supplied via the sinusoidal timestep embedding added at each residual block, which gives the network a continuous representation of "how much noise to expect."

## 3. Sampling

### 3.1 DDPM (Ancestral) Sampling

Given $\epsilon_\theta$, ancestral sampling follows Algorithm 2 of Ho et al.: starting from $x_T \sim \mathcal{N}(0,I)$, for $t = T, \dots, 1$,

$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \qquad z \sim \mathcal{N}(0,I) \text{ if } t>1 \text{ else } z=0 \tag{12}$$

with $\sigma_t^2 = \beta_t$ in this implementation. This requires a full pass through all $T$ steps and reintroduces stochastic noise at every step, which is why generation is slow: $T=1000$ sequential network evaluations are needed per batch, and the injected noise at every step means two runs with the same model and seed can diverge.

### 3.2 DDIM Sampling

Song et al. (2020) observe that the forward marginals $q(x_t \mid x_0)$ (Eq. 2) do not uniquely determine a Markovian forward process, and derive a family of non-Markovian processes sharing the same marginals but admitting a deterministic (or partially stochastic) reverse process parameterized by $\eta$. The reverse update, given a predicted noise $\epsilon_\theta(x_t, t)$, first computes a predicted clean image:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}} \tag{13}$$

clamped to the valid pixel range $[-1, 1]$ in this implementation (a practical correction; the predicted $\hat x_0$ can exceed this range early in training, where it is not yet meaningful), and then takes a step toward $x_{t-1}$ along a chosen subsequence of timesteps $\tau$:

$$x_{\tau_{i-1}} = \sqrt{\bar\alpha_{\tau_{i-1}}}\,\hat{x}_0 + \sqrt{1-\bar\alpha_{\tau_{i-1}} - \sigma_{\tau_i}^2}\;\epsilon_\theta(x_{\tau_i}, \tau_i) + \sigma_{\tau_i} z \tag{14}$$

with $\sigma_{\tau_i}^2 = \eta \cdot \tilde\beta_{\tau_i}$ (the DDPM posterior variance, Eq. 8's variance term) interpolating between fully deterministic ($\eta=0$) and the original DDPM stochasticity ($\eta=1$). Because $\tau$ can be a short subsequence of $\{1, \dots, T\}$ — for example 50 steps instead of 1000 — DDIM trades a small amount of sample quality for a large reduction in the number of network evaluations.

This is the practical motivation for evaluating both samplers in Phase 2: with $\eta=0$ and a cosine schedule, 50-step DDIM achieved a lower FID than full 1000-step DDPM on Fashion-MNIST in this implementation, in roughly $1/18$ of the wall-clock time (Section 6). This is not a universal result — it is sensitive to $n$, the noise schedule, and the dataset — but it illustrates why DDIM is the default sampler used for the conditional model in Phase 3, where inference speed matters more directly (e.g., for an interactive demo).

## 4. Text Conditioning

### 4.1 Frozen CLIP as a Text Encoder

The text-conditional model (Phase 3) introduces a caption $c$ (e.g., "a photo of a dog") and replaces $\epsilon_\theta(x_t, t)$ with $\epsilon_\theta(x_t, t, \tau(c))$, where $\tau$ is a text encoder. This implementation uses a frozen, pretrained CLIP text encoder (Radford et al. 2021, ViT-B/32) for $\tau$, rather than training a text encoder jointly with the diffusion model. Joint training of a text-image alignment from scratch typically requires orders of magnitude more data and compute than is available here (CLIP itself was trained on 400M image-text pairs); using a frozen encoder transfers the semantic structure CLIP already learned and restricts training to the U-Net's new cross-attention parameters, which is tractable on a single consumer GPU.

CLIP's output is a single pooled embedding $\tau(c) \in \mathbb{R}^{512}$, with no inherent sequence structure. Naively treating this as a single attention "token" ($n_{\text{ctx}}=1$) causes a subtle but serious failure: softmax over a single key collapses to a constant ($1.0$) regardless of the query, meaning the gradient with respect to the query and key projection matrices is identically zero — those parameters never receive a training signal even though the forward pass appears to run correctly. This implementation instead projects the 512-d embedding into $n_{\text{ctx}}=4$ "virtual tokens" of dimension $C$ (the U-Net's channel width at that block) before the cross-attention layer, so that softmax operates over a non-trivial distribution and all four attention projections ($Q$, $K$, $V$, output) receive gradients.

### 4.2 Cross-Attention

At selected U-Net resolutions (the bottleneck and the first up-sampling block in this implementation), spatial features $h \in \mathbb{R}^{B \times C \times H \times W}$ attend to the text context $\kappa \in \mathbb{R}^{B \times n_{\text{ctx}} \times C}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \tag{15}$$

with $Q$ derived from the flattened spatial features (reshaped to $B \times HW \times C$) and $K, V$ derived from $\kappa$. The result is added back to the spatial features (residual connection) before continuing through the U-Net. This is the standard cross-attention conditioning mechanism used in Stable Diffusion and Imagen, scaled down to this model's size (5.2M of 48.8M total parameters, or 10.8%, are in the cross-attention blocks).

### 4.3 Classifier-Free Guidance

Ho & Salimans (2022) note that conditional generation quality improves substantially if the model is trained to support both conditional and unconditional generation, and the two predictions are combined at sampling time. During training, the text embedding is replaced with a learned "null" embedding $\tau(\varnothing)$ — in this implementation, CLIP's encoding of the empty string — with probability $p_{\text{uncond}}$ (set to 0.1):

$$\epsilon_\theta(x_t, t, c) \to \begin{cases} \epsilon_\theta(x_t, t, \tau(\varnothing)) & \text{with probability } p_{\text{uncond}} \\ \epsilon_\theta(x_t, t, \tau(c)) & \text{otherwise} \end{cases} \tag{16}$$

At sampling time, both predictions are computed (in this implementation, batched into a single forward pass for efficiency) and combined with a guidance scale $w$:

$$\hat\epsilon = \epsilon_\theta(x_t, t, \tau(\varnothing)) + (1+w)\big(\epsilon_\theta(x_t, t, \tau(c)) - \epsilon_\theta(x_t, t, \tau(\varnothing))\big) \tag{17}$$

Equivalently written as $\hat\epsilon = (1+w)\,\epsilon_{\text{cond}} - w\,\epsilon_{\text{uncond}}$, the term being amplified is the *difference* between the conditional and unconditional predictions — that is, the part of the prediction that is specifically attributable to the caption. Setting $w=0$ recovers plain conditional sampling; increasing $w$ pushes samples further in the direction the caption implies, at the cost of reduced diversity and a tendency toward saturated, lower-entropy outputs at high $w$. This implementation uses $w=7.5$ at inference by default, consistent with values commonly used in Stable Diffusion.

## 5. Architecture Summary

| Component | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Dataset | Fashion-MNIST ($28\times28$, grayscale) | Fashion-MNIST | CIFAR-10 ($32\times32$, RGB) |
| Noise schedule | Linear | Cosine | Cosine |
| Sampler | DDPM | DDPM + DDIM | DDIM ($\eta=0$, 50 steps) + CFG |
| Conditioning | None | None | CLIP ViT-B/32, 4 virtual context tokens, cross-attention |
| Parameters | 11.6M | 11.6M | 48.8M (5.2M / 10.8% in cross-attention) |
| Training extras | — | — | EMA, linear LR warmup, gradient clipping |

## 6. Empirical Notes

FID (Heusel et al. 2017) was computed against InceptionV3 pool features, comparing 500 generated samples to the Fashion-MNIST test set (Phase 2 checkpoint, cosine schedule):

| Sampler | Steps | FID | Wall-clock time |
|---|---|---|---|
| DDPM ($\eta=0$ equivalent) | 1000 | 339.3 | 301.7s |
| DDIM ($\eta=0$) | 10 | 65.7 | 4.9s |
| DDIM ($\eta=0$) | 50 | 63.6 | 17.0s |

$n=500$ is below the $n \ge 2048$ typically recommended for stable FID estimates, so the absolute values should be read with caution, but the relative ordering — DDIM achieving substantially better FID than full DDPM at a fraction of the cost — is consistent with the DDIM paper's central claim and was the motivation for using DDIM as the default sampler in Phase 3.

## References

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.

Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML*.

Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. *ICLR*.

Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). *ICML*.

Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS Workshop*.

Heusel, M., et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (Fréchet Inception Distance). *NeurIPS*.
