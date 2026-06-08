# Draft — Dexterous Retargeting by MPPI with Covariance Matrix Adaptation

## Status: Working Draft (2026-04-25)

---

## 1. Bibliography

Complete `\thebibliography` block. Keys match `\cite{citeXX}` in the report.

### Key → Paper mapping

| Key | Paper |
|-----|-------|
| cite13 | Todorov et al., MuJoCo (2012) |
| cite26 | Pan et al., SPIDER (2025) |
| cite42 | Williams et al., MPPI — Information-Theoretic MPC (2017a) |
| cite43 | Williams et al., MPPI — Aggressive Driving (2016) |
| cite50 | Williams et al., MPPI — Theory to Parallel Computation (2017b) |
| cite51 | Busetti, Simulated Annealing temperature (2003) |
| cite58 | Abraham et al., MPPI with learned models (2020) |
| cite526 | Xue et al., DIAL-MPC (2024) |
| cite559 | Keshavarz et al., MPOPI (2025) |
| cite566 | Hansen et al., CMA-ES (2003) |
| cite613 | Yi et al., CoVO-MPC (2024) |
| cite633 | Pinneri et al., Sample-efficient CEM (2021) |

### LaTeX block

```latex
\begin{thebibliography}{99}

\bibitem{cite13}
E.~Todorov, T.~Erez, and Y.~Tassa,
``{MuJoCo}: A physics engine for model-based control,''
in \textit{Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)},
2012, pp.~5026--5033.

\bibitem{cite26}
C.~Pan, C.~Wang, H.~Qi, Z.~Liu, H.~Bharadhwaj, A.~Sharma, T.~Wu, G.~Shi,
J.~Malik, and F.~Hogan,
``{SPIDER}: Scalable physics-informed dexterous retargeting,''
\textit{arXiv preprint arXiv:2511.09484}, 2025.

\bibitem{cite42}
G.~Williams, N.~Wagener, B.~Goldfain, P.~Drews, J.~M.~Rehg, B.~Boots,
and E.~A.~Theodorou,
``Information theoretic {MPC} for model-based reinforcement learning,''
in \textit{Proc. IEEE Int. Conf. Robotics and Automation (ICRA)},
2017.

\bibitem{cite43}
G.~Williams, P.~Drews, B.~Goldfain, J.~M.~Rehg, and E.~A.~Theodorou,
``Aggressive driving with model predictive path integral control,''
in \textit{Proc. IEEE Int. Conf. Robotics and Automation (ICRA)},
2016, pp.~1433--1440.

\bibitem{cite50}
G.~Williams, A.~Aldrich, and E.~A.~Theodorou,
``Model predictive path integral control: From theory to parallel
computation,''
\textit{J. Guidance, Control, and Dynamics}, vol.~40, no.~2,
pp.~344--357, 2017.

\bibitem{cite51}
F.~Busetti,
``Simulated annealing overview,''
2003.

\bibitem{cite58}
I.~Abraham, A.~Hoque, and T.~D.~Murphey,
``Model-based generalization under parameter uncertainty using
path integral control,''
\textit{IEEE Robotics and Automation Letters}, vol.~5, no.~2,
pp.~2864--2871, 2020.

\bibitem{cite526}
H.~Xue, C.~Pan, Z.~Yi, G.~Qu, and G.~Shi,
``{DIAL-MPC}: Diffusion-inspired annealing for legged {MPC},''
\textit{arXiv preprint arXiv:2410.15898}, 2024.

\bibitem{cite559}
H.~Keshavarz, \textit{et al.},
``Control of legged robots using model predictive optimized path integral,''
\textit{arXiv preprint arXiv:2508.11917}, 2025.

\bibitem{cite566}
N.~Hansen, S.~D.~M{\"u}ller, and P.~Koumoutsakos,
``Reducing the time complexity of the derandomized evolution strategy
with covariance matrix adaptation ({CMA-ES}),''
\textit{Evolutionary Computation}, vol.~11, no.~1, pp.~1--18, 2003.

\bibitem{cite613}
Z.~Yi, C.~Pan, G.~He, G.~Qu, and G.~Shi,
``{CoVO-MPC}: Theoretical analysis of sampling-based {MPC} and optimal
covariance design,''
in \textit{Proc. Learning for Dynamics and Control (L4DC)}, 2024.

\bibitem{cite633}
C.~Pinneri, S.~Sawant, S.~Blaes, J.~Achterhold, J.~Stueckler,
M.~Rolinek, and G.~Martius,
``Sample-efficient cross-entropy method for real-time planning,''
in \textit{Proc. Conf. Robot Learning (CoRL)}, 2021.

\end{thebibliography}
```

---

## 2. Future Work — Expanded Draft

Replace the bullet-list `\section{Future work}` with this:

```latex
\section{FUTURE WORK}
\label{sec:future_work}

The experimental results reveal several open challenges that
motivate future investigation.

\paragraph{Covariance reset for closed-loop stability.}
CMA-MPPI achieves strong per-step convergence but exhibits
closed-loop instability when the cost landscape shifts between
MPC ticks (cf.\ the reward collapse on \texttt{p52-instrument}).
A promising direction is to combine CMA's directional
adaptation with a periodic covariance reset mechanism: after
every $K$ MPC steps, $\boldsymbol{\Sigma}$ is partially
reinitialized toward $\sigma^2 \mathbf{I}$, preventing the
distribution from over-specializing to a stale landscape.
Alternatively, a forgetting factor $\eta_{\text{reset}}$
could blend the learned covariance with an isotropic prior
at each step:
\begin{equation}
  \boldsymbol{\Sigma}_{t+1} = (1 - \eta_{\text{reset}})
  \boldsymbol{\Sigma}_t^{\text{CMA}} +
  \eta_{\text{reset}} \, \sigma_0^2 \mathbf{I}
\end{equation}

\paragraph{Control signal smoothing.}
All sampling-based methods produce jittery control signals
compared to gradient-based NMPC, with CMA-MPPI showing the
most pronounced oscillations. Temporal smoothing strategies
---such as exponential moving average filtering on the
control mean, or adding a control-rate penalty
$\|\mathbf{u}_t - \mathbf{u}_{t-1}\|^2$ to the cost
function---could mitigate this without sacrificing the
gradient-free advantage. The trade-off between tracking
bandwidth and actuator smoothness warrants systematic study.

\paragraph{Knot-point parameterization for scalability.}
The current implementation optimizes over the full control
sequence $\mathbf{U} \in \mathbb{R}^{mH}$, where $mH = 192
\times 320 = 61{,}440$ for the Gigahand system. This
dimensionality makes CMA's $O(d^3)$ Cholesky decomposition
a concern for longer horizons. A knot-point
parameterization---optimizing over $K \ll H$ waypoints and
interpolating the full sequence---would reduce the effective
dimension from $mH$ to $mK$, making covariance adaptation
tractable even for high-DoF systems with long horizons.

\paragraph{Weight computation: softmax vs.\ rank-based.}
Our experiments use softmax weighting for MPPI/DIAL and
rank-based weighting for CMA. The interaction between the
weight scheme and the elite selection strategy remains
underexplored. A systematic comparison---rank-based MPPI,
softmax CMA, and hybrid schemes---could reveal whether the
performance gains of CMA stem primarily from the covariance
adaptation or from the rank-based weighting itself.

\paragraph{Contact-aware sampling.}
The SPIDER pipeline provides rich contact information
(contact points, normals, forces) that is currently used
only in the cost function. Incorporating contact geometry
into the \textit{sampling distribution}---for example, by
biasing samples toward control sequences that maintain
contact at designated grasp points---could substantially
improve sample efficiency in contact-rich tasks. This
connects to the broader question of how task-specific
structure can inform the design of
$\boldsymbol{\Sigma}$~\cite{cite613}.

\paragraph{Force feedback integration.}
The current formulation operates in position/velocity space
without explicit force feedback. Integrating tactile or
force/torque sensor measurements into the MPC cost---or
using them to modulate the sampling distribution
online---would enable reactive behaviors such as slip
detection and grasp force regulation, which are essential
for real-world dexterous manipulation.
```

---

## 3. Integration checklist

- [ ] Paste `\thebibliography` block at end of `.tex`, replacing `\section{Reference}`
- [ ] Replace bullet-list `\section{Future work}` with expanded version
- [ ] Verify cite51 (Busetti) and cite58 (Abraham) are actually referenced in text; remove from bibliography if not
- [ ] cite43 = ICRA 2016 (aggressive driving), cite42 = ICRA 2017 (information-theoretic) — check each `\cite` is pointing to the right one
- [ ] Consider adding cite for MuJoCo Warp if distinct from original MuJoCo (cite13)
