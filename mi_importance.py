"""Mutual-information-based pruning importance for diffusion U-Nets.

This implements a *scalar-proxy* version of an MIPP-style criterion so that it
plugs directly into `torch_pruning`'s importance interface (one scalar per
channel; the pruner then removes the lowest-scoring fraction at a fixed
`--pruning_ratio`).

Two independent terms, each toggleable via its weight (set weight=0 to disable):

  * adjacency term  -- MIPP's principle applied locally: score a channel by how
                       much information its activation shares with the *next*
                       layer's activations (the downstream conv/linear layers
                       that consume this channel, discovered automatically from
                       torch_pruning's dependency group).

  * output term     -- the diffusion-specific idea: score a channel by how much
                       information its activation shares with the model's final
                       output (the predicted noise). Motivated by the fact that
                       diffusion U-Nets accumulate signal toward the output
                       rather than merely relaying it layer-to-layer.

Both terms use the same cheap, fully-vectorised estimator: a Gaussian
mutual-information proxy   I = -1/2 * log(1 - rho^2)   averaged over the target
dimensions, where rho is the Pearson correlation between a channel's per-sample
scalar summary and each target dimension. This is a deliberate simplification of
MIPP's predictive/TERC machinery -- it keeps the whole existing fixed-ratio,
group-coupled pipeline intact and is fast enough to run inside the calibration
loop `ddpm_prune.py` already performs. Swap in a richer estimator later if the
proxy shows signal.

Usage (see ddpm_prune.py):
    imp = MIImportance(w_output=1.0, w_adjacency=1.0)
    imp.attach(model, ignored_layers)          # register activation hooks
    for ... : out = model(noisy, t).sample     # calibration forward passes
              imp.record_output(out)
    imp.finalize()                             # concat + detach hooks
    # ... then use `imp` as the importance in tp.pruner.MagnitudePruner
"""

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
    in_names = [
        "prune_conv_in_channels",
        "prune_linear_in_channels",
    ]
    out_fns = {getattr(fn_mod, n) for n in out_names if hasattr(fn_mod, n)}
    in_fns = {getattr(fn_mod, n) for n in in_names if hasattr(fn_mod, n)}
    return out_fns, in_fns


def _channel_summary(out, module):
    """Reduce a layer output to one scalar per channel, per sample: (B, C).

    Conv2d channels live on dim 1 (B, C, H, W); Linear features live on the
    LAST dim (B, C) or (B, seq, C) inside attention. We average over every
    non-batch, non-channel dim.
    """
    if isinstance(module, nn.Conv2d):          # (B, C, H, W)
        return out.mean(dim=tuple(range(2, out.dim()))) if out.dim() > 2 else out
    # Linear: channel is the last dim
    if out.dim() == 2:                          # (B, C)
        return out
    return out.flatten(1, -2).mean(dim=1)       # (B, ..., C) -> (B, C)


def gaussian_mi_scores(A, T):
    """Gaussian-MI proxy between each column of A and the target T.

    A : (N, C)  per-sample scalar summary for each of C candidate channels
    T : (N, D)  per-sample summary of the target (downstream layer or output)
    returns (C,) : mean over the D target dims of  -1/2 log(1 - rho^2)
    """
    N = A.shape[0]
    A = A - A.mean(dim=0, keepdim=True)
    T = T - T.mean(dim=0, keepdim=True)
    A = A / (A.std(dim=0, keepdim=True) + _EPS)
    T = T / (T.std(dim=0, keepdim=True) + _EPS)
    corr = (A.t() @ T) / N                      # (C, D) Pearson correlations
    corr2 = (corr ** 2).clamp(max=1.0 - 1e-6)
    mi = -0.5 * torch.log(1.0 - corr2)          # (C, D) nats
    return mi.mean(dim=1)                        # (C,)


class MIImportance(tp.importance.Importance):
    """MIPP-style scalar-proxy importance. See module docstring."""

    def __init__(
        self,
        w_output=1.0,
        w_adjacency=1.0,
        output_pool=4,
        group_reduction="mean",
        normalizer="mean",
    ):
        self.w_output = w_output
        self.w_adjacency = w_adjacency
        self.output_pool = output_pool          # spatial pool size for the output target
        self.group_reduction = group_reduction
        self.normalizer = normalizer

        self._out_fns, self._in_fns = _prune_fn_sets()

        # activation caches (filled during calibration)
        self._buffers = {}       # module -> list[(B, C)] then finalized (N, C)
        self._handles = []
        self._output_buffer = []  # list[(B, D)] then finalized (N, D)
        self._output_target = None
        self._finalized = False

    # ------------------------------------------------------------------ capture
    def attach(self, model, ignored_layers=()):
        """Register forward hooks on every prunable Conv2d/Linear so we can
        gather per-channel activation summaries during calibration."""
        ignored = set(ignored_layers)
        for m in model.modules():
            if m in ignored:
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self._buffers[m] = []
                self._handles.append(m.register_forward_hook(self._make_hook(m)))
        return self

    def _make_hook(self, module):
        @torch.no_grad()
        def hook(mod, inp, out):
            self._buffers[module].append(_channel_summary(out, module).detach().float().cpu())
        return hook

    @torch.no_grad()
    def record_output(self, model_output):
        """Record the model's final output (predicted noise) for the output term.
        Pooled to a small grid so spatial variance is preserved cheaply."""
        if self.w_output == 0:
            return
        x = model_output.detach().float()
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, self.output_pool)   # (B, C, p, p)
        self._output_buffer.append(x.flatten(1).cpu())        # (B, C*p*p)

    def finalize(self):
        """Concatenate buffers into (N, C) tensors and remove the hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []
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

    def _magnitude_fallback(self, group, idxs):
        """Used when a group's root activations were not captured, so pruning
        still proceeds sensibly instead of erroring out."""
        for dep, gidxs in group:
            layer = dep.target.module
            if dep.handler in self._out_fns and isinstance(layer, (nn.Conv2d, nn.Linear)):
                w = layer.weight.data
                if w.dim() > 1:
                    score = w.flatten(1).abs().pow(2).sum(1)
                else:
                    score = w.abs().pow(2)
                return score[idxs]
        return torch.ones(len(idxs))

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, **kwargs):
        assert self._finalized, "call finalize() after the calibration loop"

        # locate the module whose OUT channels this group prunes (the root),
        # and the downstream consumer modules (adjacency targets)
        root_module, root_idxs = None, None
        consumers = []
        for dep, idxs in group:
            layer = dep.target.module
            if dep.handler in self._out_fns and root_module is None:
                if isinstance(layer, (nn.Conv2d, nn.Linear)) and self._buffers.get(layer) is not None:
                    root_module, root_idxs = layer, idxs
            elif dep.handler in self._in_fns:
                if self._buffers.get(layer) is not None:
                    consumers.append(layer)

        if root_module is None:
            # nothing captured for this group -> fall back to weight magnitude
            any_idxs = next(iter(group))[1]
            return self._magnitude_fallback(group, sorted(set(any_idxs)))

        root_idxs = sorted(set(root_idxs))
        A_full = self._buffers[root_module]           # (N, C_root)
        A = A_full[:, root_idxs]                       # (N, len(idxs))

        total = torch.zeros(A.shape[1])
        active_weight = 0.0

        # ---- output term: MI(channel ; final predicted noise) ----
        if self.w_output != 0 and self._output_target is not None:
            out_score = gaussian_mi_scores(A, self._output_target)
            total = total + self.w_output * self._normalize(out_score)
            active_weight += self.w_output

        # ---- adjacency term: MI(channel ; next layer activations) ----
        if self.w_adjacency != 0 and len(consumers):
            adj_scores = []
            for c in consumers:
                adj_scores.append(gaussian_mi_scores(A, self._buffers[c]))
            adj_score = torch.stack(adj_scores, dim=0).mean(dim=0)
            total = total + self.w_adjacency * self._normalize(adj_score)
            active_weight += self.w_adjacency

        if active_weight == 0:
            # both terms unavailable for this group -> magnitude fallback
            return self._magnitude_fallback(group, root_idxs)

        return total
