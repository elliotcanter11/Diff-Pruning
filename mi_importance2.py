"""Mutual-information-based pruning importance for diffusion U-Nets.

Criterion
---------
Preserve information that the network relies on at multiple downstream scales.

Three optional terms are used:

  1. NEXT-LAYER TERM
       I(A_L ; A_{L+1} | rest, t, position)

     Measures how much information a channel in layer L contributes to the
     immediately following convolutional layer.

     Because convolutions are local, samples are (image, spatial location),
     and the target is evaluated at the SAME spatial location in the consumer.

  2. INTERMEDIATE LOOKAHEAD TERM
       I(A_L ; A_{L+k} | rest, t, position)

     Measures whether information in layer L remains relevant after several
     downstream transformations.

     By default k=2, i.e. the second downstream Conv2d in forward execution
     order. This provides an intermediate scale between immediate local
     dependence and final-output dependence.

  3. OUTPUT TERM
       I(A_L ; Y | rest, t)

     Measures how much information in layer L is associated with the final
     predicted-noise output.

     A deep layer's influence on the output is spatially diffuse, so this term
     is estimated per image using pooled spatial descriptors rather than
     same-location samples.

All three terms use the same grouped Gaussian conditional-MI estimator.

Estimator
---------
For a channel/group descriptor S:

    importance = I(S ; T | rest, conditioning)

                 1/2 log(
                     det Cov(S | rest, conditioning)
                     --------------------------------
                     det Cov(S | rest, conditioning, T)
                 )

             =   1/2 log(
                     det Omega_full[S,S]
                     ------------------
                     det Omega_base[S,S]
                 )

where Omega_base is the precision matrix of [X, conditioning] and
Omega_full is the precision matrix of [X, conditioning, T].

For the local and intermediate terms, each channel is a scalar spatial sample
(block size 1). For the output term, each channel is represented by a pooled
g x g spatial descriptor, giving block size g*g.

The estimator is:
  * deterministic after calibration,
  * redundancy-aware,
  * training-free,
  * compatible with grouped channel pruning,
  * based on a joint Gaussian approximation.

The Gaussian assumption means that the estimator captures dependence through
the covariance structure. A nonparametric KSG estimate can be used as a
cross-check if this assumption becomes important.

Multi-scale criterion
---------------------
The combined importance is

    I_total =
        w_next * I_next
      + w_mid  * I_mid
      + w_output * I_output.

Setting any weight to zero disables that term and its associated capture/
computation.

For example:

    w_next=1.0
    w_mid=0.5
    w_output=1.0

gives a criterion emphasizing both immediate and eventual information while
giving the intermediate horizon somewhat less weight.

The intermediate lookahead distance is controlled by:

    mid_lookahead=2

which means the second downstream Conv2d in forward execution order.

Only Conv2d layers participate in MI scoring. Attention/linear groups fall
back to weight magnitude.

A cheap overlap diagnostic records how much the final MI ranking agrees with
weight magnitude.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp


_EPS = 1e-8


# ---------------------------------------------------------------------------
# Torch-Pruning helpers
# ---------------------------------------------------------------------------

def _prune_fn_sets():
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

    out_fns = {
        getattr(fn_mod, n)
        for n in out_names
        if hasattr(fn_mod, n)
    }

    in_fns = {
        getattr(fn_mod, n)
        for n in in_names
        if hasattr(fn_mod, n)
    }

    return out_fns, in_fns


# ---------------------------------------------------------------------------
# Conditioning features
# ---------------------------------------------------------------------------

def _timestep_embedding(t_norm, num_freqs=4):
    """Small Fourier feature embedding of normalized diffusion timestep."""
    feats = [t_norm.unsqueeze(1)]

    for k in range(num_freqs):
        freq = (2.0 ** k) * math.pi

        feats.append(
            torch.sin(freq * t_norm).unsqueeze(1)
        )

        feats.append(
            torch.cos(freq * t_norm).unsqueeze(1)
        )

    return torch.cat(feats, dim=1)


def _loc_embedding(coords):
    """
    coords: (M, 2) normalized (u, v)

    Returns a small smooth positional basis so spatial trends do not
    masquerade as channel dependence.
    """
    u = coords[:, 0]
    v = coords[:, 1]

    return torch.stack(
        [
            u,
            v,
            torch.sin(math.pi * u),
            torch.cos(math.pi * u),
            torch.sin(math.pi * v),
            torch.cos(math.pi * v),
        ],
        dim=1,
    )


def _zscore(Z):
    if Z.shape[1] == 0:
        return Z

    return (
        Z - Z.mean(0, keepdim=True)
    ) / (
        Z.std(0, keepdim=True) + _EPS
    )


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------

def _ridge_inv(Z, shrink):
    """
    Compute a ridge-stabilized inverse covariance matrix.

    This helper is retained for compatibility with the previous implementation.
    """
    Zc = Z - Z.mean(0, keepdim=True)

    S = (Zc.t() @ Zc) / (Z.shape[0] - 1)

    d = S.shape[0]

    tr = torch.diagonal(S).mean()

    S = (
        S
        + shrink
        * tr
        * torch.eye(
            d,
            device=S.device,
            dtype=S.dtype,
        )
    )

    return torch.linalg.inv(S)


def _diag_blocks(Omega, C, b):
    """
    Extract the C per-channel b x b diagonal blocks of the X-part of Omega.

    Omega contains the precision matrix for:

        [channel blocks | conditioning | target]

    Only the channel portion is considered here.
    """
    ar = torch.arange(C, device=Omega.device)
    ab = torch.arange(b, device=Omega.device)

    r = (
        ar.view(C, 1, 1) * b
        + ab.view(1, b, 1)
    )

    c = (
        ar.view(C, 1, 1) * b
        + ab.view(1, 1, b)
    )

    return Omega[r, c]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()

    ra = (ra - ra.mean()) / (ra.std() + _EPS)
    rb = (rb - rb.mean()) / (rb.std() + _EPS)

    return float((ra * rb).mean())


# ---------------------------------------------------------------------------
# MI importance
# ---------------------------------------------------------------------------

class MIImportance(tp.importance.Importance):
    """
    Grouped Gaussian conditional-MI importance for structured pruning.

    The criterion combines three optional information horizons:

        next:
            I(A_L ; A_{L+1})

        intermediate:
            I(A_L ; A_{L+k})

        output:
            I(A_L ; Y)

    where k is controlled by `mid_lookahead`.

    Parameters
    ----------
    w_output : float
        Weight of final-output MI term.

    w_adjacency : float
        Weight of immediate-next-layer MI term.

    w_intermediate : float
        Weight of intermediate lookahead MI term.

    mid_lookahead : int
        Number of downstream Conv2d layers used for the intermediate target.

        2 means:
            root -> next Conv2d -> intermediate Conv2d

        3 means:
            root -> next Conv2d -> Conv2d -> intermediate Conv2d

        Values <= 1 are rejected because the intermediate term should be
        distinct from the immediate adjacency term.

    num_locations : int
        Number of spatial coordinates sampled per image for local/intermediate
        terms.

    out_grid : int
        Per-channel spatial grid size for the output term.

    out_target_pool : int
        Spatial pooling size applied to the final predicted-noise target.

    shrinkage : float
        Ridge shrinkage applied before covariance inversion.

    target_dim_cap : int
        Maximum number of target dimensions used for local/intermediate terms.

    num_freqs : int
        Number of Fourier frequencies used for timestep conditioning.

    prune_ratio : float
        Ratio used only for the MI-vs-magnitude diagnostic.

    normalizer : str
        "mean" divides importance by its mean.
    """

    def __init__(
        self,
        w_output=1.0,
        w_adjacency=1.0,
        w_intermediate=0.5,
        mid_lookahead=2,
        num_locations=4,
        out_grid=1,
        out_target_pool=8,
        shrinkage=1e-2,
        target_dim_cap=512,
        num_freqs=4,
        prune_ratio=0.0,
        normalizer="mean",
    ):
        if mid_lookahead <= 1:
            raise ValueError(
                "mid_lookahead must be >= 2 so the intermediate term "
                "is distinct from the immediate adjacency term."
            )

        self.w_output = w_output
        self.w_adjacency = w_adjacency
        self.w_intermediate = w_intermediate

        self.mid_lookahead = mid_lookahead

        self.L = num_locations
        self.g = out_grid
        self.out_pool = out_target_pool

        self.shrinkage = shrinkage
        self.target_dim_cap = target_dim_cap
        self.num_freqs = num_freqs

        self.prune_ratio = prune_ratio
        self.normalizer = normalizer

        self._out_fns, self._in_fns = _prune_fn_sets()

        self.device = torch.device("cpu")

        # ------------------------------------------------------------------
        # Conv layer bookkeeping
        # ------------------------------------------------------------------

        self._convs = set()

        # Ordered by module traversal / forward execution structure.
        #
        # This ordering is used only to define the intermediate lookahead
        # target. Immediate consumers are still determined from the
        # Torch-Pruning dependency graph.
        self._conv_order = []
        self._conv_index = {}

        # ------------------------------------------------------------------
        # Activation capture
        # ------------------------------------------------------------------

        # Immediate / intermediate spatial samples:
        #
        #   module -> list[(B*L, C)]
        #
        self._loc_buf = {}

        # Final-output descriptors:
        #
        #   module -> list[(B, C*g*g)]
        #
        self._img_buf = {}

        self._handles = []

        self._coords = None

        # Conditioning
        self._coords_buf = []
        self._loc_t_buf = []

        self._img_t_buf = []

        self._loc_cond = None
        self._img_cond = None

        # Final output target
        self._out_buf = []
        self._out_target = None

        self._finalized = False

        # Diagnostic:
        #
        #   (n_channels, jaccard, spearman)
        #
        self._diag = []

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def attach(self, model, ignored_layers=()):
        """
        Attach forward hooks to all participating Conv2d layers.

        The Conv2d order is recorded so that the intermediate target can be
        defined by forward-order distance.
        """
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            pass

        ignored = set(ignored_layers)

        self._conv_order = []
        self._conv_index = {}

        for m in model.modules():
            if (
                m in ignored
                or not isinstance(m, nn.Conv2d)
            ):
                continue

            self._convs.add(m)

            self._conv_index[m] = len(self._conv_order)
            self._conv_order.append(m)

            self._loc_buf[m] = []
            self._img_buf[m] = []

            self._handles.append(
                m.register_forward_hook(
                    self._make_hook(m)
                )
            )

        return self

    def new_pass(self, batch_size):
        """
        Start a new calibration forward pass.

        The same normalized spatial coordinates are used across all Conv2d
        layers so that local/intermediate target rows remain aligned.
        """
        self._coords = torch.rand(
            batch_size,
            self.L,
            2,
        )

    def _gather(self, feat):
        """
        Sample each feature map at the common normalized coordinates.

        feat:
            (B, C, H, W)

        Returns:
            (B*L, C)

        Each row corresponds to one (image, spatial location) pair.
        """
        B, C, H, W = feat.shape

        coords = self._coords[:B].to(feat.device)

        ys = (
            coords[..., 0] * H
        ).long().clamp_(0, H - 1)

        xs = (
            coords[..., 1] * W
        ).long().clamp_(0, W - 1)

        idx = (
            ys * W + xs
        ).unsqueeze(1).expand(
            B,
            C,
            -1,
        )

        g = feat.reshape(
            B,
            C,
            H * W,
        ).gather(
            2,
            idx,
        )

        return g.permute(
            0,
            2,
            1,
        ).reshape(
            B * self.L,
            C,
        )

    def _make_hook(self, module):
        @torch.no_grad()
        def hook(mod, inp, out):
            if (
                out.dim() != 4
                or self._coords is None
            ):
                return

            # Immediate + intermediate terms use the same spatial samples.
            if (
                self.w_adjacency != 0
                or self.w_intermediate != 0
            ):
                self._loc_buf[module].append(
                    self._gather(out)
                    .detach()
                    .half()
                    .cpu()
                )

            # Output term uses whole-image pooled descriptors.
            if self.w_output != 0:
                d = F.adaptive_avg_pool2d(
                    out,
                    self.g,
                )

                self._img_buf[module].append(
                    d.reshape(
                        d.shape[0],
                        -1,
                    )
                    .detach()
                    .half()
                    .cpu()
                )

        return hook

    @torch.no_grad()
    def record_output(self, model_output):
        """
        Record the final predicted-noise output for the output MI term.
        """
        if (
            self.w_output == 0
            or model_output.dim() != 4
        ):
            return

        d = F.adaptive_avg_pool2d(
            model_output.detach().float(),
            self.out_pool,
        )

        self._out_buf.append(
            d.reshape(
                d.shape[0],
                -1,
            ).cpu()
        )

    @torch.no_grad()
    def record_timesteps(self, timesteps):
        """
        Record timestep conditioning.

        Local/intermediate terms are aligned with B*L spatial samples.
        Output term is aligned with B image samples.
        """
        t = (
            timesteps
            .detach()
            .reshape(-1)
            .float()
            .cpu()
        )

        if (
            self.w_adjacency != 0
            or self.w_intermediate != 0
        ):
            self._loc_t_buf.append(
                t.repeat_interleave(self.L)
            )

            self._coords_buf.append(
                self._coords.reshape(-1, 2).cpu()
            )

        if self.w_output != 0:
            self._img_t_buf.append(t)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self):
        """
        Remove hooks and concatenate calibration buffers.
        """
        for h in self._handles:
            h.remove()

        self._handles = []

        all_t = (
            self._loc_t_buf
            + self._img_t_buf
        )

        t_max = (
            float(torch.cat(all_t).max()) + 1.0
            if len(all_t)
            else 1.0
        )

        # --------------------------------------------------------------
        # Spatial terms
        # --------------------------------------------------------------

        if (
            (
                self.w_adjacency != 0
                or self.w_intermediate != 0
            )
            and len(self._loc_t_buf)
        ):
            lt = (
                torch.cat(self._loc_t_buf)
                / t_max
            )

            coords = torch.cat(
                self._coords_buf
            )

            self._loc_cond = torch.cat(
                [
                    _timestep_embedding(
                        lt,
                        self.num_freqs,
                    ),
                    _loc_embedding(coords),
                ],
                dim=1,
            )

            for m, ch in self._loc_buf.items():
                self._loc_buf[m] = (
                    torch.cat(ch, 0)
                    if len(ch)
                    else None
                )

        # --------------------------------------------------------------
        # Output term
        # --------------------------------------------------------------

        if (
            self.w_output != 0
            and len(self._img_t_buf)
        ):
            it = (
                torch.cat(self._img_t_buf)
                / t_max
            )

            self._img_cond = _timestep_embedding(
                it,
                self.num_freqs,
            )

            if len(self._out_buf):
                self._out_target = torch.cat(
                    self._out_buf,
                    0,
                )
            else:
                self._out_target = None

            for m, ch in self._img_buf.items():
                self._img_buf[m] = (
                    torch.cat(ch, 0)
                    if len(ch)
                    else None
                )

        self._finalized = True

        return self

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """
        Clear captured activations while retaining diagnostic history.

        Used when recalibrating after a pruning step.
        """
        for h in self._handles:
            h.remove()

        self._handles = []

        self._convs = set()

        self._conv_order = []
        self._conv_index = {}

        self._loc_buf = {}
        self._img_buf = {}

        self._coords_buf = []
        self._loc_t_buf = []

        self._img_t_buf = []

        self._out_buf = []

        self._loc_cond = None
        self._img_cond = None
        self._out_target = None

        self._coords = None

        self._finalized = False

        return self

    # ------------------------------------------------------------------
    # Layer relationships
    # ------------------------------------------------------------------

    def _get_intermediate_target(self, root):
        """
        Return the Conv2d that is `mid_lookahead` positions downstream from
        root in Conv2d execution/module order.

        Example with mid_lookahead=2:

            root -> Conv A -> Conv B

        returns Conv B.

        Returns None if there are not enough downstream Conv2d layers.
        """
        if root not in self._conv_index:
            return None

        i = self._conv_index[root]

        j = i + self.mid_lookahead

        if j >= len(self._conv_order):
            return None

        return self._conv_order[j]

    def _get_immediate_consumers(self, group):
        """
        Get Conv2d layers that directly consume the root activation according
        to the Torch-Pruning dependency graph.
        """
        consumers = []

        for dep, _ in group:
            layer = dep.target.module

            if (
                dep.handler in self._in_fns
                and layer in self._convs
            ):
                consumers.append(layer)

        # Preserve order but remove duplicates.
        unique = []

        for layer in consumers:
            if layer not in unique:
                unique.append(layer)

        return unique

    # ------------------------------------------------------------------
    # Covariance preparation
    # ------------------------------------------------------------------

    def _normalize(self, imp):
        if self.normalizer == "mean":
            return imp / (
                imp.mean() + _EPS
            )

        return imp

    def _prep_cov(self, Xz, cond, T):
        """
        Full covariance of:

            [Xz | cond | T]

        The covariance is computed once for a term. The greedy loop then
        extracts submatrices after channels are removed.
        """
        Z = torch.cat(
            [
                Xz,
                cond,
                T,
            ],
            dim=1,
        )

        Zc = (
            Z - Z.mean(
                0,
                keepdim=True,
            )
        )

        Cov = (
            Zc.t() @ Zc
        ) / (
            Z.shape[0] - 1
        )

        return {
            "Cov": Cov,
            "Cb": Xz.shape[1],
            "E": cond.shape[1],
            "D": T.shape[1],
        }

    def _sub_inv(self, Cov, idx):
        """
        Invert a selected covariance submatrix with ridge stabilization.
        """
        S = (
            Cov.index_select(0, idx)
            .index_select(1, idx)
        )

        d = S.shape[0]

        tr = torch.diagonal(S).mean()

        eye = torch.eye(
            d,
            device=S.device,
            dtype=S.dtype,
        )

        S = (
            S
            + self.shrinkage
            * tr
            * eye
        )

        return torch.linalg.inv(S)

    # ------------------------------------------------------------------
    # Greedy conditional-MI elimination
    # ------------------------------------------------------------------

    def _greedy(self, terms, C):
        """
        True one-at-a-time greedy elimination.

        At each iteration, evaluate the conditional MI contributed by every
        surviving channel given all other currently surviving channels.

        The channel with the smallest combined MI is removed.

        This is redundancy-aware: if two channels are highly redundant, the
        first one can be removed cheaply, while the remaining copy becomes
        important once the first is gone.

        The resulting importance is the removal order:

            removed first -> lowest importance
            removed last  -> highest importance
        """
        dev = self.device

        kept = list(range(C))

        imp = torch.zeros(
            C,
            device=dev,
        )

        order = 0.0

        while len(kept) > 1:
            kt = torch.tensor(
                kept,
                device=dev,
            )

            combined = torch.zeros(
                len(kept),
                device=dev,
            )

            # ----------------------------------------------------------
            # Evaluate every enabled MI horizon
            # ----------------------------------------------------------

            for term in terms:
                b = term["block"]
                Cb = term["Cb"]
                E = term["E"]
                D = term["D"]
                Cov = term["Cov"]

                xcols = (
                    kt.view(-1, 1) * b
                    + torch.arange(
                        b,
                        device=dev,
                    )
                ).reshape(-1)

                cond_c = torch.arange(
                    Cb,
                    Cb + E,
                    device=dev,
                )

                t_c = torch.arange(
                    Cb + E,
                    Cb + E + D,
                    device=dev,
                )

                base_idx = torch.cat(
                    [
                        xcols,
                        cond_c,
                    ]
                )

                full_idx = torch.cat(
                    [
                        base_idx,
                        t_c,
                    ]
                )

                nk = len(kept)

                # Precision conditioned only on surviving X + conditioning.
                base_precision = self._sub_inv(
                    Cov,
                    base_idx,
                )

                # Precision conditioned on surviving X + conditioning + T.
                full_precision = self._sub_inv(
                    Cov,
                    full_idx,
                )

                base = torch.linalg.slogdet(
                    _diag_blocks(
                        base_precision,
                        nk,
                        b,
                    )
                )[1]

                full = torch.linalg.slogdet(
                    _diag_blocks(
                        full_precision,
                        nk,
                        b,
                    )
                )[1]

                cmi = (
                    0.5
                    * (full - base)
                ).clamp(
                    min=0.0
                )

                combined = (
                    combined
                    + term["w"]
                    * self._normalize(cmi)
                )

            # Lowest combined MI is pruned first.
            j = int(
                torch.argmin(combined)
            )

            imp[kept[j]] = order

            kept.pop(j)

            order += 1.0

        # Last surviving channel gets the highest importance.
        imp[kept[0]] = order

        return self._normalize(imp)

    # ------------------------------------------------------------------
    # Magnitude fallback
    # ------------------------------------------------------------------

    def _mag(self, module, idxs):
        w = module.weight.data

        if w.dim() > 1:
            s = w.flatten(1).norm(
                p=2,
                dim=1,
            )
        else:
            s = w.abs()

        return s[idxs].to(
            self.device
        )

    def _magnitude_fallback(self, group, idxs):
        for dep, _ in group:
            layer = dep.target.module

            if (
                dep.handler in self._out_fns
                and isinstance(
                    layer,
                    (nn.Conv2d, nn.Linear),
                )
            ):
                return self._mag(
                    layer,
                    idxs,
                )

        return torch.ones(
            len(idxs),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------

    def _record_overlap(
        self,
        root,
        idxs,
        mi_imp,
    ):
        if self.prune_ratio <= 0:
            return

        mag = self._mag(
            root,
            idxs,
        )

        k = max(
            1,
            int(
                round(
                    len(idxs)
                    * self.prune_ratio
                )
            ),
        )

        mi_cut = set(
            mi_imp.argsort()[:k].tolist()
        )

        mag_cut = set(
            mag.argsort()[:k].tolist()
        )

        jac = (
            len(mi_cut & mag_cut)
            /
            len(mi_cut | mag_cut)
        )

        self._diag.append(
            (
                len(idxs),
                jac,
                _spearman(
                    mi_imp,
                    mag,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Main importance call
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        group,
        ch_groups=1,
        **kwargs,
    ):
        assert self._finalized, (
            "call finalize() after the calibration loop"
        )

        root = None
        root_idxs = None

        # --------------------------------------------------------------
        # Identify root Conv2d
        # --------------------------------------------------------------

        for dep, idxs in group:
            layer = dep.target.module

            if (
                dep.handler in self._out_fns
                and root is None
                and layer in self._convs
            ):
                root = layer
                root_idxs = idxs

        if root is None:
            fallback_idxs = sorted(
                set(
                    next(
                        iter(group)
                    )[1]
                )
            )

            return self._magnitude_fallback(
                group,
                fallback_idxs,
            )

        root_idxs = sorted(
            set(root_idxs)
        )

        # --------------------------------------------------------------
        # Determine downstream targets
        # --------------------------------------------------------------

        immediate_consumers = []

        if self.w_adjacency != 0:
            immediate_consumers = (
                self._get_immediate_consumers(
                    group
                )
            )

        intermediate_target = None

        if self.w_intermediate != 0:
            intermediate_target = (
                self._get_intermediate_target(
                    root
                )
            )

        dev = self.device

        try:
            terms = []

            # ==========================================================
            # 1. IMMEDIATE NEXT-LAYER TERM
            # ==========================================================

            if (
                self.w_adjacency != 0
                and immediate_consumers
                and self._loc_buf.get(root)
                is not None
            ):
                Xz = _zscore(
                    self._loc_buf[root][
                        :,
                        root_idxs,
                    ]
                    .float()
                    .to(dev)
                )

                target_parts = []

                for consumer in immediate_consumers:
                    target = self._loc_buf.get(
                        consumer
                    )

                    if target is not None:
                        target_parts.append(
                            target
                        )

                if target_parts:
                    T = torch.cat(
                        target_parts,
                        dim=1,
                    ).float().to(dev)

                    # Prevent the target from becoming too large.
                    if (
                        T.shape[1]
                        > self.target_dim_cap
                    ):
                        perm = torch.randperm(
                            T.shape[1],
                            device=dev,
                        )[
                            :self.target_dim_cap
                        ]

                        T = T[:, perm]

                    t = self._prep_cov(
                        Xz,
                        _zscore(
                            self._loc_cond.to(dev)
                        ),
                        _zscore(T),
                    )

                    t["w"] = (
                        self.w_adjacency
                    )

                    # One scalar spatial sample per channel.
                    t["block"] = 1

                    terms.append(t)

            # ==========================================================
            # 2. INTERMEDIATE LOOKAHEAD TERM
            # ==========================================================

            if (
                self.w_intermediate != 0
                and intermediate_target is not None
                and self._loc_buf.get(root)
                is not None
                and self._loc_buf.get(
                    intermediate_target
                )
                is not None
            ):
                Xz = _zscore(
                    self._loc_buf[root][
                        :,
                        root_idxs,
                    ]
                    .float()
                    .to(dev)
                )

                T = (
                    self._loc_buf[
                        intermediate_target
                    ]
                    .float()
                    .to(dev)
                )

                # Same target-dimension cap as the immediate term.
                if (
                    T.shape[1]
                    > self.target_dim_cap
                ):
                    perm = torch.randperm(
                        T.shape[1],
                        device=dev,
                    )[
                        :self.target_dim_cap
                    ]

                    T = T[:, perm]

                t = self._prep_cov(
                    Xz,
                    _zscore(
                        self._loc_cond.to(dev)
                    ),
                    _zscore(T),
                )

                t["w"] = (
                    self.w_intermediate
                )

                # Same scalar spatial descriptor as the adjacency term.
                t["block"] = 1

                terms.append(t)

            # ==========================================================
            # 3. FINAL OUTPUT TERM
            # ==========================================================

            if (
                self.w_output != 0
                and self._img_buf.get(root)
                is not None
                and self._out_target is not None
            ):
                b = self.g * self.g

                root_idx_tensor = torch.tensor(
                    root_idxs,
                    device=dev,
                )

                cols = (
                    root_idx_tensor.view(-1, 1)
                    * b
                    + torch.arange(
                        b,
                        device=dev,
                    )
                ).reshape(-1)

                Xz = _zscore(
                    self._img_buf[root][
                        :,
                        cols,
                    ]
                    .float()
                    .to(dev)
                )

                t = self._prep_cov(
                    Xz,
                    _zscore(
                        self._img_cond.to(dev)
                    ),
                    _zscore(
                        self._out_target
                        .float()
                        .to(dev)
                    ),
                )

                t["w"] = self.w_output

                # Each channel is represented by g*g pooled values.
                t["block"] = b

                terms.append(t)

            # ==========================================================
            # FALLBACK
            # ==========================================================

            if not terms:
                return self._magnitude_fallback(
                    group,
                    root_idxs,
                )

            # ==========================================================
            # COMBINED GREEDY MI
            # ==========================================================

            imp = self._greedy(
                terms,
                len(root_idxs),
            )

            self._record_overlap(
                root,
                root_idxs,
                imp,
            )

            return imp

        except Exception as e:
            print(
                "[MIImportance] group -> "
                f"magnitude fallback "
                f"({type(e).__name__}: {e})"
            )

            return self._magnitude_fallback(
                group,
                root_idxs,
            )

    # ------------------------------------------------------------------
    # Diagnostic reporting
    # ------------------------------------------------------------------

    def report_diagnostic(self):
        if not self._diag:
            print(
                "[MIImportance] "
                "no diagnostic recorded"
            )
            return

        n = sum(
            c
            for c, _, _ in self._diag
        )

        jac = (
            sum(
                c * j
                for c, j, _
                in self._diag
            )
            / n
        )

        spr = (
            sum(
                c * s
                for c, _, s
                in self._diag
            )
            / n
        )

        print(
            "\n[MIImportance] overlap vs magnitude "
            f"(weighted over {len(self._diag)} "
            f"groups, {n} channels):"
        )

        print(
            "    pruned-set Jaccard = "
            f"{jac:.3f}   "
            "(1.0 = identical cuts to magnitude)"
        )

        print(
            "    Spearman rank corr = "
            f"{spr:.3f}   "
            "(1.0 = identical ranking)"
        )