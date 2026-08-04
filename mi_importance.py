"""Mutual-information-based pruning importance for diffusion U-Nets.

Criterion: keep the channels that preserve the mutual information the network
relies on; remove the channels whose removal costs the least MI. Two terms, each
toggleable via its weight (0 disables it and skips its capture/compute):

  * adjacency term  -- I(A_L ; A_{L+1}) between a layer and the NEXT layer.
                       Convolutions are LOCAL, so a region of L most affects the
                       same region of L+1. We therefore estimate this per spatial
                       location: samples are (image, pixel), a channel is its raw
                       value at that pixel, the target is the consumer layer's
                       values at the SAME location. Conditioned on the other
                       channels, the timestep t, AND the spatial position (so a
                       position trend can't masquerade as dependence).

  * output term     -- I(A_L ; Y) between a layer and the final output (predicted
                       noise). A deep layer's effect on the output is spatially
                       diffuse, NOT local, so this term is WHOLE-LAYER, per image:
                       samples are images, a channel is a coarse gxg pooled
                       descriptor of its map, the target is the whole (pooled)
                       output. Conditioned on the other channels and t.

Estimator -- closed-form GAUSSIAN CONDITIONAL mutual information (grouped).

  Model [X, conditioning, T] as jointly Gaussian. For a channel whose descriptor
  is a block S of columns, the MI it contributes on top of all the other channels
  (and the conditioning) is, in closed form,

        importance = I(S ; T | rest) = 1/2 * log( det Cov(S | rest)
                                                 / det Cov(S | rest, T) )
                   = 1/2 * log( det Omega_full[S,S] / det Omega_base[S,S] )

  where Omega_base = precision of [X, cond], Omega_full = precision of
  [X, cond, T], and Omega[S,S] is S's diagonal block. For a scalar channel
  (adjacency, block size 1) this reduces to the ratio of precision diagonals.
  It is deterministic, needs no training, is redundancy-aware (conditioning on the
  other channels), and scales to any width. The one assumption is joint
  Gaussianity -- dependence up to second order; use a KSG cross-check if that bites.

Only Conv2d layers participate (the bulk of a DDPM U-Net); attention/linear groups
fall back to weight magnitude. Still one scalar per channel, so it drops into
torch_pruning's fixed-ratio, group-coupled interface unchanged.

A cheap overlap diagnostic (no extra training) records, per group, how much the MI
ranking agrees with weight-magnitude -- printed at the end via report_diagnostic().
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp

_EPS = 1e-8


def _prune_fn_sets():
    fn_mod = tp.pruner.function
    out_names = ["prune_conv_out_channels", "prune_linear_out_channels",
                 "prune_batchnorm_out_channels", "prune_groupnorm_out_channels"]
    in_names = ["prune_conv_in_channels", "prune_linear_in_channels"]
    out_fns = {getattr(fn_mod, n) for n in out_names if hasattr(fn_mod, n)}
    in_fns = {getattr(fn_mod, n) for n in in_names if hasattr(fn_mod, n)}
    return out_fns, in_fns


def _timestep_embedding(t_norm, num_freqs=4):
    feats = [t_norm.unsqueeze(1)]
    for k in range(num_freqs):
        freq = (2.0 ** k) * math.pi
        feats.append(torch.sin(freq * t_norm).unsqueeze(1))
        feats.append(torch.cos(freq * t_norm).unsqueeze(1))
    return torch.cat(feats, dim=1)


def _loc_embedding(coords):
    """coords: (M, 2) normalised (u, v) -> (M, 6), a small smooth position basis
    so the adjacency MI can control for spatial trends."""
    u, v = coords[:, 0], coords[:, 1]
    return torch.stack([u, v,
                        torch.sin(math.pi * u), torch.cos(math.pi * u),
                        torch.sin(math.pi * v), torch.cos(math.pi * v)], dim=1)


def _zscore(Z):
    if Z.shape[1] == 0:
        return Z
    return (Z - Z.mean(0, keepdim=True)) / (Z.std(0, keepdim=True) + _EPS)


def _ridge_inv(Z, shrink):
    """inv(cov(Z)) with a little ridge shrinkage for numerical stability."""
    Zc = Z - Z.mean(0, keepdim=True)
    S = (Zc.t() @ Zc) / (Z.shape[0] - 1)
    d = S.shape[0]
    tr = torch.diagonal(S).mean()
    S = S + shrink * tr * torch.eye(d, device=S.device, dtype=S.dtype)
    return torch.linalg.inv(S)


def _diag_blocks(Omega, C, b):
    """Extract the C per-channel bxb diagonal blocks of the X-part of Omega."""
    ar = torch.arange(C, device=Omega.device)
    ab = torch.arange(b, device=Omega.device)
    r = ar.view(C, 1, 1) * b + ab.view(1, b, 1)
    c = ar.view(C, 1, 1) * b + ab.view(1, 1, b)
    return Omega[r, c]                     # (C, b, b)


def _spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + _EPS)
    rb = (rb - rb.mean()) / (rb.std() + _EPS)
    return float((ra * rb).mean())


class MIImportance(tp.importance.Importance):
    """Grouped Gaussian conditional-MI importance. See module docstring."""

    def __init__(self, w_output=1.0, w_adjacency=1.0,
                 num_locations=4, out_grid=1, out_target_pool=8,
                 shrinkage=1e-2, target_dim_cap=512, num_freqs=4,
                 prune_ratio=0.0, normalizer="mean"):
        self.w_output = w_output
        self.w_adjacency = w_adjacency
        self.L = num_locations
        self.g = out_grid                     # per-channel gxg descriptor (output term)
        self.out_pool = out_target_pool       # output target pooled to this grid
        self.shrinkage = shrinkage
        self.target_dim_cap = target_dim_cap
        self.num_freqs = num_freqs
        self.prune_ratio = prune_ratio
        self.normalizer = normalizer

        self._out_fns, self._in_fns = _prune_fn_sets()
        self.device = torch.device("cpu")
        self._convs = set()

        # capture buffers
        self._loc_buf = {}     # conv -> (M_loc, C)     per-(image,location) samples
        self._img_buf = {}     # conv -> (N_img, C*g*g) per-image pooled descriptors
        self._handles = []
        self._coords = None
        self._coords_buf, self._loc_t_buf, self._img_t_buf, self._out_buf = [], [], [], []
        self._loc_cond = self._img_cond = self._out_target = None
        self._finalized = False

        self._diag = []        # (n_channels, jaccard, spearman) per group

    # ------------------------------------------------------------------ capture
    def attach(self, model, ignored_layers=()):
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            pass
        ignored = set(ignored_layers)
        for m in model.modules():
            if m in ignored or not isinstance(m, nn.Conv2d):
                continue
            self._convs.add(m)
            self._loc_buf[m], self._img_buf[m] = [], []
            self._handles.append(m.register_forward_hook(self._make_hook(m)))
        return self

    def new_pass(self, batch_size):
        self._coords = torch.rand(batch_size, self.L, 2)

    def _gather(self, feat):
        """feat (B,C,H,W) -> (B*L, C) sampled at self._coords (same normalised
        coords across every layer, so all buffers are row-aligned)."""
        B, C, H, W = feat.shape
        coords = self._coords[:B].to(feat.device)
        ys = (coords[..., 0] * H).long().clamp_(0, H - 1)
        xs = (coords[..., 1] * W).long().clamp_(0, W - 1)
        idx = (ys * W + xs).unsqueeze(1).expand(B, C, -1)
        g = feat.reshape(B, C, H * W).gather(2, idx)          # (B, C, L)
        return g.permute(0, 2, 1).reshape(B * self.L, C)

    def _make_hook(self, module):
        @torch.no_grad()
        def hook(mod, inp, out):
            if out.dim() != 4 or self._coords is None:
                return
            if self.w_adjacency != 0:
                self._loc_buf[module].append(self._gather(out).detach().half().cpu())
            if self.w_output != 0:
                d = F.adaptive_avg_pool2d(out, self.g)         # (B, C, g, g)
                self._img_buf[module].append(d.reshape(d.shape[0], -1).detach().half().cpu())
        return hook

    @torch.no_grad()
    def record_output(self, model_output):
        if self.w_output == 0 or model_output.dim() != 4:
            return
        d = F.adaptive_avg_pool2d(model_output.detach().float(), self.out_pool)
        self._out_buf.append(d.reshape(d.shape[0], -1).cpu())   # (B, 3*pool*pool)

    @torch.no_grad()
    def record_timesteps(self, timesteps):
        t = timesteps.detach().reshape(-1).float().cpu()
        if self.w_adjacency != 0:
            self._loc_t_buf.append(t.repeat_interleave(self.L))
            self._coords_buf.append(self._coords.reshape(-1, 2).cpu())
        if self.w_output != 0:
            self._img_t_buf.append(t)

    def finalize(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        # common timestep normaliser
        all_t = self._loc_t_buf + self._img_t_buf
        t_max = float(torch.cat(all_t).max()) + 1.0 if len(all_t) else 1.0

        if self.w_adjacency != 0 and len(self._loc_t_buf):
            lt = torch.cat(self._loc_t_buf) / t_max
            coords = torch.cat(self._coords_buf)
            self._loc_cond = torch.cat([_timestep_embedding(lt, self.num_freqs),
                                        _loc_embedding(coords)], dim=1)
            for m, ch in self._loc_buf.items():
                self._loc_buf[m] = torch.cat(ch, 0) if len(ch) else None
        if self.w_output != 0 and len(self._img_t_buf):
            it = torch.cat(self._img_t_buf) / t_max
            self._img_cond = _timestep_embedding(it, self.num_freqs)
            self._out_target = torch.cat(self._out_buf, 0)
            for m, ch in self._img_buf.items():
                self._img_buf[m] = torch.cat(ch, 0) if len(ch) else None
        self._finalized = True
        return self

    def reset(self):
        """Clear captured activations (but keep the diagnostic log) so the object
        can be re-attached and re-calibrated on the current model -- used for
        greedy/iterative pruning, where MI is recomputed on the surviving
        channels after each pruning step."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self._convs = set()
        self._loc_buf, self._img_buf = {}, {}
        self._coords_buf, self._loc_t_buf, self._img_t_buf, self._out_buf = [], [], [], []
        self._loc_cond = self._img_cond = self._out_target = None
        self._coords = None
        self._finalized = False
        return self

    # ----------------------------------------------------------------- scoring
    def _normalize(self, imp):
        if self.normalizer == "mean":
            return imp / (imp.mean() + _EPS)
        return imp

    def _prep_cov(self, Xz, cond, T):
        """Full covariance of [Xz (channel blocks) | cond | T], computed ONCE.
        The greedy loop then just slices sub-covariances of it -- so removing a
        channel is cheap (drop its rows/cols) and never re-reads the samples."""
        Z = torch.cat([Xz, cond, T], dim=1)
        Zc = Z - Z.mean(0, keepdim=True)
        Cov = (Zc.t() @ Zc) / (Z.shape[0] - 1)
        return {'Cov': Cov, 'Cb': Xz.shape[1], 'E': cond.shape[1], 'D': T.shape[1]}

    def _sub_inv(self, Cov, idx):
        S = Cov.index_select(0, idx).index_select(1, idx)
        d = S.shape[0]
        tr = torch.diagonal(S).mean()
        return torch.linalg.inv(S + self.shrinkage * tr * torch.eye(d, device=S.device, dtype=S.dtype))

    def _greedy(self, terms, C):
        """True one-at-a-time greedy elimination (the redundancy-correct form,
        i.e. TERC): repeatedly drop the channel with the LOWEST conditional MI
        given the channels still kept, recomputing after each removal -- so a
        channel stops looking redundant the moment its duplicate is gone, and
        exactly one of a redundant pair survives. Importance = removal order
        (removed first = least important = pruned first)."""
        dev = self.device
        kept = list(range(C))
        imp = torch.zeros(C, device=dev)
        order = 0.0
        while len(kept) > 1:
            kt = torch.tensor(kept, device=dev)
            combined = torch.zeros(len(kept), device=dev)
            for t in terms:
                b, Cb, E, D, Cov = t['block'], t['Cb'], t['E'], t['D'], t['Cov']
                xcols = (kt.view(-1, 1) * b + torch.arange(b, device=dev)).reshape(-1)
                cond_c = torch.arange(Cb, Cb + E, device=dev)
                t_c = torch.arange(Cb + E, Cb + E + D, device=dev)
                base_idx = torch.cat([xcols, cond_c])
                full_idx = torch.cat([base_idx, t_c])
                nk = len(kept)
                base = torch.linalg.slogdet(_diag_blocks(self._sub_inv(Cov, base_idx), nk, b))[1]
                full = torch.linalg.slogdet(_diag_blocks(self._sub_inv(Cov, full_idx), nk, b))[1]
                cmi = (0.5 * (full - base)).clamp(min=0.0)
                combined = combined + t['w'] * self._normalize(cmi)
            j = int(torch.argmin(combined))
            imp[kept[j]] = order
            kept.pop(j)
            order += 1.0
        imp[kept[0]] = order
        return self._normalize(imp)

    def _mag(self, module, idxs):
        w = module.weight.data
        s = w.flatten(1).norm(p=2, dim=1) if w.dim() > 1 else w.abs()
        return s[idxs].to(self.device)

    def _magnitude_fallback(self, group, idxs):
        for dep, _ in group:
            layer = dep.target.module
            if dep.handler in self._out_fns and isinstance(layer, (nn.Conv2d, nn.Linear)):
                return self._mag(layer, idxs)
        return torch.ones(len(idxs), device=self.device)

    def _record_overlap(self, root, idxs, mi_imp):
        if self.prune_ratio <= 0:
            return
        mag = self._mag(root, idxs)
        k = max(1, int(round(len(idxs) * self.prune_ratio)))
        mi_cut = set(mi_imp.argsort()[:k].tolist())        # lowest-MI = pruned
        mag_cut = set(mag.argsort()[:k].tolist())
        jac = len(mi_cut & mag_cut) / len(mi_cut | mag_cut)
        self._diag.append((len(idxs), jac, _spearman(mi_imp, mag)))

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, **kwargs):
        assert self._finalized, "call finalize() after the calibration loop"
        root, root_idxs, consumers = None, None, []
        for dep, idxs in group:
            layer = dep.target.module
            if dep.handler in self._out_fns and root is None and layer in self._convs:
                root, root_idxs = layer, idxs
            elif dep.handler in self._in_fns and layer in self._convs:
                consumers.append(layer)
        if root is None:
            return self._magnitude_fallback(group, sorted(set(next(iter(group))[1])))

        root_idxs = sorted(set(root_idxs))
        dev = self.device
        try:
            terms = []
            if self.w_adjacency != 0 and consumers and self._loc_buf.get(root) is not None:
                Xz = _zscore(self._loc_buf[root][:, root_idxs].float().to(dev))
                T = torch.cat([self._loc_buf[c] for c in consumers], 1).float().to(dev)
                if T.shape[1] > self.target_dim_cap:
                    T = T[:, torch.randperm(T.shape[1], device=dev)[: self.target_dim_cap]]
                t = self._prep_cov(Xz, _zscore(self._loc_cond.to(dev)), _zscore(T))
                t['w'], t['block'] = self.w_adjacency, 1
                terms.append(t)

            if self.w_output != 0 and self._img_buf.get(root) is not None and self._out_target is not None:
                b = self.g * self.g
                cols = (torch.tensor(root_idxs).view(-1, 1) * b + torch.arange(b)).reshape(-1)
                Xz = _zscore(self._img_buf[root][:, cols].float().to(dev))
                t = self._prep_cov(Xz, _zscore(self._img_cond.to(dev)),
                                   _zscore(self._out_target.float().to(dev)))
                t['w'], t['block'] = self.w_output, b
                terms.append(t)

            if not terms:
                return self._magnitude_fallback(group, root_idxs)
            imp = self._greedy(terms, len(root_idxs))
            self._record_overlap(root, root_idxs, imp)
            return imp
        except Exception as e:
            print(f"[MIImportance] group -> magnitude fallback ({type(e).__name__}: {e})")
            return self._magnitude_fallback(group, root_idxs)

    def report_diagnostic(self):
        if not self._diag:
            print("[MIImportance] no diagnostic recorded")
            return
        n = sum(c for c, _, _ in self._diag)
        jac = sum(c * j for c, j, _ in self._diag) / n
        spr = sum(c * s for c, _, s in self._diag) / n
        print("\n[MIImportance] overlap vs magnitude "
              f"(weighted over {len(self._diag)} groups, {n} channels):")
        print(f"    pruned-set Jaccard = {jac:.3f}   (1.0 = identical cuts to magnitude)")
        print(f"    Spearman rank corr = {spr:.3f}   (1.0 = identical ranking)")
