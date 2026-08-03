"""Mutual-information-based pruning importance for diffusion U-Nets.

Criterion: keep the channels that preserve the mutual information the network
relies on, remove the channels whose removal costs the least MI. Two targets,
each toggleable via its weight (0 disables it):

  * adjacency term  -- preserve I(A_L ; A_{L+1}): MI between a layer's channels
                       and the NEXT layer's (its consumer convs') activations.
  * output term     -- preserve I(A_L ; Y): MI between a layer's channels and the
                       model's final output (the predicted noise). Motivated by
                       diffusion U-Nets accumulating signal toward the output.

Estimator -- closed-form GAUSSIAN CONDITIONAL mutual information.

  Each conv channel is treated as a scalar random variable sampled at spatial
  LOCATIONS (per image x per location), so a channel keeps its full activation
  distribution -- no lossy "energy" summary. For a layer with channel matrix
  X (M samples x C channels) and target T (M x D), modelling [X, t, T] as jointly
  Gaussian gives, in closed form, the MI that each channel contributes on top of
  all the others (and conditioned on the noise level t):

        importance(i) = I(X_i ; T | X_{-i}, t)
                      = 1/2 * log( Var(X_i | X_{-i}, t) / Var(X_i | X_{-i}, t, T) )
                      = 1/2 * log( diag(inv(cov[X,t,T]))_i / diag(inv(cov[X,t]))_i )

  This is a standard MI computation (Gaussian graphical models / Gaussian
  information bottleneck): deterministic, no training, scales to any layer width
  (only needs covariance matrices), and is redundancy-aware by construction --
  a channel that merely duplicates information its neighbours already carry
  contributes ~0 and is pruned. The one assumption is joint Gaussianity, i.e. it
  captures dependence up to second order; use a KSG cross-check if that matters.

  t (the diffusion timestep) is included in the always-conditioned block via a
  small sinusoidal embedding, so importances are computed at fixed noise level.

Only Conv2d layers participate (the bulk of a DDPM U-Net); attention/linear
groups fall back to weight magnitude. Still a scalar-per-channel importance, so
it drops into torch_pruning's fixed-ratio, group-coupled interface unchanged.

Usage (see ddpm_prune.py):
    imp = MIImportance(w_output=1.0, w_adjacency=1.0)
    imp.attach(model, ignored_layers)
    for ...:
        imp.new_pass(batch_size)                   # sample per-image locations
        out = model(noisy, t).sample               # hooks gather at those locs
        imp.record_output(out); imp.record_timesteps(t)
    imp.finalize()
    # ... then use `imp` as the importance in tp.pruner.MagnitudePruner
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp

_EPS = 1e-8


def _prune_fn_sets():
    """Return (out_channel_fns, in_channel_fns) from whatever this
    torch_pruning version exposes, guarding against renamed/missing names."""
    fn_mod = tp.pruner.function
    out_names = [
        "prune_conv_out_channels",
        "prune_linear_out_channels",
        "prune_batchnorm_out_channels",
        "prune_groupnorm_out_channels",
    ]
    in_names = ["prune_conv_in_channels", "prune_linear_in_channels"]
    out_fns = {getattr(fn_mod, n) for n in out_names if hasattr(fn_mod, n)}
    in_fns = {getattr(fn_mod, n) for n in in_names if hasattr(fn_mod, n)}
    return out_fns, in_fns


def _timestep_embedding(t_norm, num_freqs=4):
    """Sinusoidal embedding of the normalised timestep. (M,) -> (M, 1+2*freqs)."""
    feats = [t_norm.unsqueeze(1)]
    for k in range(num_freqs):
        freq = (2.0 ** k) * math.pi
        feats.append(torch.sin(freq * t_norm).unsqueeze(1))
        feats.append(torch.cos(freq * t_norm).unsqueeze(1))
    return torch.cat(feats, dim=1)


def _zscore(Z):
    return (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + _EPS)


def _cov(Z):
    Zc = Z - Z.mean(0, keepdim=True)
    return (Zc.t() @ Zc) / (Z.shape[0] - 1)


def _inv_diag_ridged(Z, shrinkage):
    """Diagonal of the inverse covariance (precision) of columns of Z, with a
    little ridge shrinkage for numerical stability."""
    S = _cov(Z)
    d = S.shape[0]
    tr = torch.diagonal(S).mean()
    S = S + shrinkage * tr * torch.eye(d, device=S.device, dtype=S.dtype)
    return torch.diagonal(torch.linalg.inv(S))


class MIImportance(tp.importance.Importance):
    """Closed-form Gaussian conditional-MI importance. See module docstring."""

    def __init__(
        self,
        w_output=1.0,
        w_adjacency=1.0,
        num_locations=8,
        shrinkage=1e-2,
        target_dim_cap=512,
        num_freqs=4,
        normalizer="mean",
    ):
        self.w_output = w_output
        self.w_adjacency = w_adjacency
        self.num_locations = num_locations
        self.shrinkage = shrinkage
        self.target_dim_cap = target_dim_cap
        self.num_freqs = num_freqs
        self.normalizer = normalizer

        self._out_fns, self._in_fns = _prune_fn_sets()
        self.device = torch.device("cpu")

        self._buffers = {}          # conv module -> list[(B*L, C)] then (M, C)
        self._handles = []
        self._output_buffer = []     # list[(B*L, 3)] then (M, 3)
        self._output_target = None
        self._timestep_buffer = []   # list[(B*L,)]
        self._t_feats = None         # (M, E) timestep embedding
        self._coords = None          # (B, L, 2) normalised locations for the pass
        self._finalized = False

    # ------------------------------------------------------------------ capture
    def attach(self, model, ignored_layers=()):
        """Hook every prunable Conv2d so we can gather per-location channel
        samples during calibration."""
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            pass
        ignored = set(ignored_layers)
        for m in model.modules():
            if m in ignored:
                continue
            if isinstance(m, nn.Conv2d):
                self._buffers[m] = []
                self._handles.append(m.register_forward_hook(self._make_hook(m)))
        return self

    def new_pass(self, batch_size):
        """Sample a fresh set of normalised spatial locations for this pass; every
        hooked layer (and the output) is read at the SAME normalised locations, so
        all samples are aligned by row across layers."""
        self._coords = torch.rand(batch_size, self.num_locations, 2)

    def _gather(self, feat):
        """feat: (B, C, H, W) -> (B*L, C) sampled at self._coords."""
        B, C, H, W = feat.shape
        coords = self._coords[:B].to(feat.device)
        ys = (coords[..., 0] * H).long().clamp_(0, H - 1)     # (B, L)
        xs = (coords[..., 1] * W).long().clamp_(0, W - 1)
        flat_idx = (ys * W + xs).unsqueeze(1).expand(B, C, -1)  # (B, C, L)
        gathered = feat.reshape(B, C, H * W).gather(2, flat_idx)  # (B, C, L)
        return gathered.permute(0, 2, 1).reshape(B * self.num_locations, C)

    def _make_hook(self, module):
        @torch.no_grad()
        def hook(mod, inp, out):
            if out.dim() != 4 or self._coords is None:
                return
            self._buffers[module].append(self._gather(out).detach().half().cpu())
        return hook

    @torch.no_grad()
    def record_output(self, model_output):
        """Sample the final output (predicted noise) at the same locations."""
        if self.w_output == 0 or self._coords is None:
            return
        x = model_output.detach().float()
        if x.dim() == 4:
            self._output_buffer.append(self._gather(x).half().cpu())   # (B*L, 3)

    @torch.no_grad()
    def record_timesteps(self, timesteps):
        """Record per-sample timestep (broadcast to each location of the image)."""
        t = timesteps.detach().reshape(-1).repeat_interleave(self.num_locations)
        self._timestep_buffer.append(t.float().cpu())

    def finalize(self):
        for h in self._handles:
            h.remove()
        self._handles = []

        if len(self._timestep_buffer):
            t = torch.cat(self._timestep_buffer, dim=0)
            t_norm = t / (float(t.max().item()) + 1.0)
            self._t_feats = _timestep_embedding(t_norm, self.num_freqs)
        self._timestep_buffer = []

        for m, chunks in self._buffers.items():
            self._buffers[m] = torch.cat(chunks, dim=0) if len(chunks) else None
        if len(self._output_buffer):
            self._output_target = torch.cat(self._output_buffer, dim=0)
        self._output_buffer = []
        self._finalized = True
        return self

    # ----------------------------------------------------------------- scoring
    def _normalize(self, imp):
        if self.normalizer is None:
            return imp
        if self.normalizer == "mean":
            return imp / (imp.mean() + _EPS)
        if self.normalizer == "sum":
            return imp / (imp.sum() + _EPS)
        if self.normalizer == "max":
            return imp / (imp.max() + _EPS)
        if self.normalizer == "standarization":
            return (imp - imp.min()) / (imp.max() - imp.min() + _EPS)
        raise NotImplementedError(self.normalizer)

    def _prep_target(self, T):
        T = _zscore(T.float().to(self.device))
        if T.shape[1] > self.target_dim_cap:
            sel = torch.randperm(T.shape[1], device=T.device)[: self.target_dim_cap]
            T = T[:, sel]
        return T

    def _magnitude_fallback(self, group, idxs):
        for dep, gidxs in group:
            layer = dep.target.module
            if dep.handler in self._out_fns and isinstance(layer, (nn.Conv2d, nn.Linear)):
                w = layer.weight.data
                score = w.flatten(1).abs().pow(2).sum(1) if w.dim() > 1 else w.abs().pow(2)
                return score[idxs].to(self.device)
        return torch.ones(len(idxs), device=self.device)

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, **kwargs):
        assert self._finalized, "call finalize() after the calibration loop"

        # find the conv whose OUT channels this group prunes, and its consumers
        root_module, root_idxs, consumers = None, None, []
        for dep, idxs in group:
            layer = dep.target.module
            if dep.handler in self._out_fns and root_module is None:
                if isinstance(layer, nn.Conv2d) and self._buffers.get(layer) is not None:
                    root_module, root_idxs = layer, idxs
            elif dep.handler in self._in_fns and self._buffers.get(layer) is not None:
                consumers.append(layer)

        if root_module is None:                       # e.g. attention/linear group
            any_idxs = sorted(set(next(iter(group))[1]))
            return self._magnitude_fallback(group, any_idxs)

        root_idxs = sorted(set(root_idxs))
        try:
            X = _zscore(self._buffers[root_module][:, root_idxs].float().to(self.device))
            C = X.shape[1]
            tf = self._t_feats.to(self.device) if self._t_feats is not None else X.new_zeros(X.shape[0], 0)
            tf = _zscore(tf) if tf.shape[1] > 0 else tf
            cond = torch.cat([X, tf], dim=1)                       # base conditioning block
            base_prec = _inv_diag_ridged(cond, self.shrinkage)[:C]  # 1/Var(X_i | rest)

            total = torch.zeros(C, device=self.device)
            active = 0.0

            if self.w_output != 0 and self._output_target is not None:
                T = self._prep_target(self._output_target)
                full_prec = _inv_diag_ridged(torch.cat([cond, T], dim=1), self.shrinkage)[:C]
                imp = (0.5 * torch.log((full_prec / base_prec).clamp(min=1.0)))
                total = total + self.w_output * self._normalize(imp)
                active += self.w_output

            if self.w_adjacency != 0 and len(consumers):
                T = self._prep_target(torch.cat([self._buffers[c] for c in consumers], dim=1))
                full_prec = _inv_diag_ridged(torch.cat([cond, T], dim=1), self.shrinkage)[:C]
                imp = (0.5 * torch.log((full_prec / base_prec).clamp(min=1.0)))
                total = total + self.w_adjacency * self._normalize(imp)
                active += self.w_adjacency

            if active == 0:
                return self._magnitude_fallback(group, root_idxs)
            return total
        except Exception as e:                        # numerical / shape issues -> safe fallback
            print(f"[MIImportance] group fell back to magnitude ({type(e).__name__}: {e})")
            return self._magnitude_fallback(group, root_idxs)
