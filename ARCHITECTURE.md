# Molecular PFM mainline

`etflow/models/pfm_model.py` uses the native ET-Flow SO(3) backbone in `models/model.py` and `networks/torchmd_net/`. A shared scalar head predicts one bounded log-variance per atom, repeated across its three coordinates. The velocity field is centered per molecule. Input features, bond/radius graph extension and the SO(3) equivariant representation are retained.

For `P = I - 11ᵀ/N`, covariance is `Gamma_v = P diag(g) P` on the centered coordinate subspace (with an identity over Cartesian coordinates). `pfm_probability.py` implements its subspace Gaussian NLL, covariance products, projected noise and marginal variances without adding artificial variance in translation directions. Projected NLL is normalized by `3(N-1)` and omits parameter-independent constants.

`pfm_sampling.py` integrates either `v` or the corrected Ito drift `v + lambda(t)/2 * (div_H Gamma_v + Gamma_v score)`, with noise `sqrt(lambda(t)) P diag(sqrt(g)) dW`. The independent-Gaussian-source score is `(t*v-x)/(source_scale²*(1-t))`. Covariance divergence uses atomwise Rademacher VJPs, projecting before and after the sign product. Exact derivatives and JVP/VJP duality are checked in tests. No integration-step graph is retained.

The main config explicitly selects `lambda(t)=0.3*(1-t)`; constant mode preserves earlier API behavior. This is a documented schedule addition to align the released sampling option with the manuscript. Both schedules scale drift and noise consistently. Endpoint readouts are `mean_x=x+(1-t)*v`, `Sigma_x=(1-t)²*Gamma_v`; generation completes with the endpoint mean at a finite terminal time.

`data/` preserves native featurization and deterministic per-epoch conformer selection. `cli/train.py` uses Lightning, AdamW and one warmup/cosine schedule over the configured run. `last.ckpt` resumes the same completed epoch with optimizer/scheduler/RNG states; there is no special multi-stage training path.

`commons/pfm_artifacts.py` records data/config/checkpoint/source identities, commits each molecule atomically and rejects output collisions or incomplete/corrupt summaries. `cli/evaluate.py` retains native symmetry-aware heavy-atom RMSD and COV/MAT. `cli/chirality.py` keeps raw generation separate from input-stereo correction. `cli/diagnostics.py` learns bins and the time-only baseline on validation reference states, then evaluates test residual moments and NLL without using test residuals to fit the baseline.
