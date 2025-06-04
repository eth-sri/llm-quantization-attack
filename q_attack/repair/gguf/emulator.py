import torch
from tqdm import tqdm

from q_attack.repair.gguf.pygguf_dequantize import GGUFData, Q2KData, Q3KData, Q4KData, Q5KData, Q6KData

class GGUFEmulator:
    """Each Emulator must have:
        `_register_params(self)` and `quantize(self)`
    """
    def __init__(
        self,
        x: torch.Tensor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        self.x = x.to(device).half().float()  # half().float() is necessary
        # self.x = x.to(device).float()  # for emulating
        self.x_exact = x.to(device) # for computing interval
        dim0, dim1, dim2 = x.shape
        self.num_superblocks = dim0
        self.num_blocks = dim1
        self.blocksize = dim2
        self.device = device
        self.GROUP_MAX_EPS = 1e-30

        # [0]*num_blocks + [1]*num_blocks + ... + [num_superblocks-1]*num_blocks
        self.sb_repeated_id = torch.arange(self.num_superblocks).view(-1, 1).expand(-1, self.num_blocks).reshape(-1)
        # [0, 1, ..., num_blocks-1]*num_superblocks
        self.b_repeated_id = torch.arange(self.num_blocks).view(1, -1).expand(self.num_superblocks, -1).reshape(-1)

        self._register_params(kwargs)

    def register_params(self):
        # use for registering some hyperparameters (e.g., nmax)
        raise NotImplementedError

    def quantize(self) -> GGUFData:
        raise NotImplementedError

    def dequantize_torch(self) -> torch.Tensor:
        assert hasattr(self, "quant_data"), f"make sure self.quantize() registers self.quant_data"
        quant_data: GGUFData = self.quant_data
        deq = torch.from_numpy(quant_data.dequantize()).to(self.device)
        return deq

    def get_width(
        self,
        unfreeze_block: bool = False,
        unfreeze_maxmin: bool = False,
        freeze_sensitive_iters: int = 0,
    ):
        """
        • define the range between x and dequantized_value as interval

        • freeze (i) argmax(scale, min) block and (ii) argmax(min) weights
        """
        max_each_block, argmax_each_block = self.x_exact.max(dim=-1)
        min_each_block, argmin_each_block = self.x_exact.min(dim=-1)

        box_min = self.x_exact.clone()
        box_max = self.x_exact.clone()
        deq = self.dequantize_torch().to(self.x_exact.dtype)
        diff = deq - self.x_exact
        box_min[diff < 0] = deq[diff < 0]
        box_max[diff > 0] = deq[diff > 0]

        is_freeze = torch.zeros(self.num_superblocks, self.num_blocks, self.blocksize, dtype=torch.bool)

        type_with_shift = Q245KEmulator
        type_without_shift = Q3KEmulator | Q6KEmulator

        # trick1: to fix d & dmin, set all params in argmax(scale, min) to zero
        if not unfreeze_block:
            if isinstance(self, type_with_shift):
                assert (
                    hasattr(self, "argmax_scales") and hasattr(self, "argmax_mins")
                ), "make sure self.quantize() registers self.argmax_scales and self.argmax_mins"

                is_freeze[torch.arange(self.num_superblocks), self.argmax_scales.view(-1), :] = 1
                is_freeze[torch.arange(self.num_superblocks), self.argmax_mins.view(-1), :] = 1
            elif isinstance(self, type_without_shift):
                assert hasattr(self, "argabsmax_scales")
                is_freeze[torch.arange(self.num_superblocks), self.argabsmax_scales.view(-1), :] = 1
            else:
                raise NotImplementedError

        # trick2: set argmaxmin(x) of each block to zero
        if not unfreeze_maxmin:
            is_freeze[self.sb_repeated_id, self.b_repeated_id, argmax_each_block.view(-1)] = 1
            is_freeze[self.sb_repeated_id, self.b_repeated_id, argmin_each_block.view(-1)] = 1

        box_min[is_freeze] = self.x_exact[is_freeze]
        box_max[is_freeze] = self.x_exact[is_freeze]

        if freeze_sensitive_iters > 0:
            # add noise within the box, and freeze the block whose dequantized value changes.
            for _ in tqdm(range(freeze_sensitive_iters), desc="Freezing sensitive weights", leave=False):
                alpha = torch.rand(box_min.shape).to(self.device)
                w_noised = box_min + alpha * (box_max - box_min)
                temp_instance = self.__class__(w_noised, num_bit=self.num_bit)
                temp_instance.quantize()
                deq_noised = temp_instance.dequantize_torch().to(self.x_exact.dtype)
                diff_noised = deq_noised - deq
                is_freeze[diff_noised != 0] = 1
                box_min[is_freeze] = self.x_exact[is_freeze]
                box_max[is_freeze] = self.x_exact[is_freeze]

        self.box_min = box_min
        self.box_max = box_max

        # statistics
        width = box_max - box_min
        self.width = width
        self.nonzero = width.ne(0).sum().item() / width.numel()
        self.nonzero_average = width[width.ne(0)].mean().item()

        return self.box_min, self.box_max

class Q245KEmulator(GGUFEmulator):
    """explanation"""
    ThisData = Q2KData | Q4KData | Q5KData  # TODO: Q2K
    def __init__(self, x: torch.Tensor, **kwargs):
        self.num_bit = kwargs.get("num_bit", None)

        super().__init__(x, **kwargs)

        # Q4_K, Q5_K have the same structure
        if self.num_bit == 2:
            assert self.num_blocks == 16
            assert self.blocksize == 16
        else:
            assert self.num_blocks == 8
            assert self.blocksize == 32

    def _register_params(self, kwargs):
        # somehow rdelta and nmax have to be in the form of torch.tensor
        # double_quant_max is wrapped with tensor in double_quant
        if self.num_bit == 2:
            self.nstep = kwargs.get("nstep", 15)
            self.rmin = kwargs.get("rmin", -0.5)
            self.rdelta = torch.tensor(kwargs.get("rdelta", 0.1)).to(self.device)
            self.nmax = torch.tensor(kwargs.get("nmax", 3)).to(self.device)
            self.double_quant_max = kwargs.get("double_quant_max", 15)
        elif self.num_bit == 4:
            self.nstep = kwargs.get("nstep", 20)
            self.rmin = kwargs.get("rmin", -1.0)
            self.rdelta = torch.tensor(kwargs.get("rdelta", 0.1)).to(self.device)
            self.nmax = torch.tensor(kwargs.get("nmax", 15))
            self.double_quant_max = kwargs.get("double_quant_max", 63)
        elif self.num_bit == 5:
            self.nstep = kwargs.get("nstep", 15)
            self.rmin = kwargs.get("rmin", -0.5)
            self.rdelta = torch.tensor(kwargs.get("rdelta", 0.1)).to(self.device)
            self.nmax = torch.tensor(kwargs.get("nmax", 31))
            self.double_quant_max = kwargs.get("double_quant_max", 63)
        else:
            raise NotImplementedError(f"Only 2, 4, 5 bits are supported but got {self.num_bit}")

    def quantize(self) -> ThisData:
        with torch.no_grad():
            self._make_qkx2_quants()
            self._double_quant()
            self._final_quant()
        if self.num_bit == 2:
            class_name = Q2KData
        elif self.num_bit == 4:
            class_name = Q4KData
        elif self.num_bit == 5:
            class_name = Q5KData
        else:
            raise NotImplementedError("Only 4, 5 bits are supported.")
        quant_data = class_name(
            scale_factors=self.d.detach().cpu().float().numpy().reshape(self.num_superblocks, 1, 1),
            scale_offsets=self.dmin.detach().cpu().float().numpy().reshape(self.num_superblocks, 1, 1),
            quantized_factors=self.lscales.detach().cpu().to(torch.uint8).numpy().reshape(self.num_superblocks, self.num_blocks, 1),
            quantized_offsets=self.lmins.detach().cpu().to(torch.uint8).numpy().reshape(self.num_superblocks, self.num_blocks, 1),
            qs=self.L.detach().cpu().to(torch.uint8).numpy(),
        )
        self.quant_data = quant_data
        return quant_data

    def _make_qkx2_quants(self):
        # 1: weights
        sum_x2 = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
        weights = torch.zeros(self.num_superblocks, self.num_blocks, self.blocksize).to(self.device)
        if self.num_bit == 2:
            for j in range(self.num_blocks):
                for l in range(self.blocksize):
                    weights[:, j, l] = torch.abs(self.x[:, j, l])
        else:
            for j in range(self.num_blocks):
                for l in range(self.blocksize):
                    sum_x2[:, j] += self.x[:, j, l] * self.x[:, j, l]
                av_x = torch.sqrt(sum_x2[:, j] / self.blocksize)
                for l in range(self.blocksize):
                    weights[:, j, l] = av_x + torch.abs(self.x[:, j, l])

        # 2: make_qkx2_quants
        sum_w = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
        sum_x = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
        min_ = torch.min(self.x, dim=-1).values
        min_ = torch.where(min_ > 0, torch.zeros_like(min_), min_)
        max_ = torch.max(self.x, dim=-1).values
        zero_scale_mask = max_ == min_
        for i in range(self.blocksize):
            sum_w += weights[:, :, i]
            sum_x += weights[:, :, i] * self.x[:, :, i]


        iscale: torch.Tensor = self.nmax / (max_ - min_)
        scale = 1 / iscale
        best_mad = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)  # but use squared error
        L = torch.zeros(self.num_superblocks, self.num_blocks, self.blocksize).to(self.device)
        # 2-1: baseline
        for i in range(self.blocksize):
            l = torch.round(iscale * (self.x[:, :, i] - min_))
            L[:, :, i] = torch.clamp(l, 0, self.nmax)
            diff = scale * L[:, :, i] + min_ - self.x[:, :, i]
            if self.num_bit == 2:
                diff = torch.abs(diff) # use_mad
            else:
                diff = diff * diff
            best_mad += weights[:, :, i] * diff

        # self.debug_mad = best_mad
        # 2-2: grid search
        # self.debug_D = dict()
        for is_ in range(self.nstep + 1):
            iscale = (self.rmin + self.rdelta * is_ + self.nmax) / (max_ - min_)
            sum_l = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
            sum_l2 = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
            sum_xl = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
            Laux = torch.zeros(self.num_superblocks, self.num_blocks, self.blocksize).to(self.device)
            for i in range(self.blocksize):
                l = torch.round(iscale * (self.x[:, :, i] - min_))
                l = torch.clamp(l, 0, self.nmax)
                Laux[:, :, i] = l
                sum_l += weights[:, :, i] * l
                sum_l2 += weights[:, :, i] * l * l
                sum_xl += weights[:, :, i] * l * self.x[:, :, i]
            D = sum_w * sum_l2 - sum_l * sum_l
            # self.debug_D[is_] = D
            # D=NaN happens when max_ == min_ (e.g., for untrained embeddings). In this case, scale = 0 and L = 0 are set later.
            # assert torch.all(D >= 0), f"|D<0| = {(D<0).sum()} ({(D<0).sum() / D.numel() * 100:.2f})%, min={D.min():.2e}"

            this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
            msk_min_pos = this_min > 0
            this_min[msk_min_pos] = 0
            this_scale[msk_min_pos] = sum_xl[msk_min_pos] / sum_l2[msk_min_pos]

            mad = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
            for i in range(self.blocksize):
                diff = this_scale * Laux[:, :, i] + this_min - self.x[:, :, i]
                if self.num_bit == 2:
                    diff = torch.abs(diff) # use_mad
                else:
                    diff = diff * diff
                mad += weights[:, :, i] * diff

            mask = mad < best_mad
            for i in range(self.blocksize):
                L[:, :, i][mask] = Laux[:, :, i][mask]
            best_mad[mask] = mad[mask]
            scale[mask] = this_scale[mask]
            min_[mask] = this_min[mask]

            scale[zero_scale_mask] = 0
            L[zero_scale_mask] = 0

        self.min_ = - min_
        self.scale = scale
        self.L = L

        assert torch.all(self.min_ >= 0)
        assert torch.all(self.scale >= 0)
        assert torch.all(self.L >= 0) and torch.all(self.L <= self.nmax)

    def _double_quant(self):
        max_scale, argmax_scale = self.scale.max(dim=-1)
        max_min, argmax_min = self.min_.max(dim=-1)
        # wrap with torch.tensor seems necessary
        dq_scale = torch.tensor(self.double_quant_max, dtype=torch.float).to(self.device)

        inv_scale = dq_scale / max_scale
        inv_min = dq_scale / max_min
        lscales = torch.zeros(self.num_superblocks, self.num_blocks, dtype=torch.uint8).to(self.device)
        lmins = torch.zeros(self.num_superblocks, self.num_blocks, dtype=torch.uint8).to(self.device)
        for i in range(self.num_blocks):
            ls = torch.round(inv_scale * self.scale[:, i])
            lm = torch.round(inv_min * self.min_[:, i])
            ls = torch.clamp(ls, 0, int(self.double_quant_max))
            lm = torch.clamp(lm, 0, int(self.double_quant_max))
            lscales[:, i] = ls
            lmins[:, i] = lm

        self.d = (max_scale / dq_scale).half()
        self.dmin = (max_min / dq_scale).half()
        self.argmax_scales = argmax_scale
        self.argmax_mins = argmax_min
        self.lscales = lscales
        self.lmins = lmins

    def _final_quant(self):
        for i in range(self.num_blocks):
            sc = self.lscales[:, i]
            d = self.d.float() * sc
            mask = d != 0
            d_nonzero = torch.where(d != 0, d, torch.ones_like(d))
            m = self.lmins[:, i]
            dm = self.dmin.float() * m
            for j in range(self.blocksize):
                l = torch.round((self.x[:, i, j] + dm) / d_nonzero)
                l = torch.clamp(l, 0, self.nmax)
                self.L[:, i, j][mask] = l[mask]


class Q3KEmulator(GGUFEmulator):
    """explanation"""
    ThisData = Q3KData
    def __init__(self, x: torch.Tensor, **kwargs):
        self.num_bit = 3
        super().__init__(x, **kwargs)

        assert self.num_blocks == 16
        assert self.blocksize == 16

    def _register_params(self, kwargs):
        self.nmax = kwargs.get("nmax", 4)

    def quantize(self) -> ThisData:
        with torch.no_grad():
            self._make_q3_quants()
            self._double_quant()
            self._final_quant()
        assert torch.all(self.L >= 0) & torch.all(self.L <= 7), f"{self.L.min()}, {self.L.max()}"
        assert torch.all(self.lscales >= 0) & torch.all(self.lscales <= 63), f"{self.lscales.min()}, {self.lscales.max()}"
        quant_data = Q3KData(
            scale_factors=self.d.detach().cpu().float().numpy().reshape(self.num_superblocks, 1, 1),
            quantized_factors=self.lscales.detach().cpu().to(torch.int8).numpy().reshape(self.num_superblocks, self.num_blocks, 1) - 32,
            qs=self.L.detach().cpu().to(torch.int8).numpy() - 4,
        )
        self.quant_data = quant_data
        return quant_data

    def _make_q3_quants(self):
        # amax = torch.max(self.x.abs(), dim=-1).values
        argabsmax = torch.argmax(self.x.abs(), dim=-1)
        max_ = torch.gather(self.x, dim=-1, index=argabsmax.unsqueeze(-1)).squeeze(-1)
        # can be zero (e.g., for untrained embeddings)
        # assert torch.all(max_ != 0), f"{(max_ == 0).sum()} {(max_ == 0).sum() / max_.numel() * 100:.2f}%"
        # mask_zero_scale = torch.abs(max_) < self.GROUP_MAX_EPS
        mask_zero_scale = max_ == 0    # this also works?
        iscale = -self.nmax / max_
        # self.debug_max_ = max_
        # self.debug_iscale = iscale
        sumlx = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
        suml2 = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
        L = torch.zeros(self.num_superblocks, self.num_blocks, self.blocksize).to(self.device)
        for i in range(self.blocksize):
            l = torch.round(iscale * self.x[:, :, i])
            l = torch.clamp(l, -self.nmax, self.nmax - 1)
            L[:, :, i] = l
            w = self.x[:, :, i] * self.x[:, :, i]
            sumlx += w * self.x[:, :, i] * l
            suml2 += w * l * l

        # self.debug_sumlx = sumlx.clone()
        # self.debug_suml2 = suml2.clone()

        for itry in range(5):
            n_changed = torch.zeros(self.num_superblocks, self.num_blocks, dtype=torch.int8).to(self.device)
            still_changeable = torch.ones(self.num_superblocks, self.num_blocks, dtype=torch.bool).to(self.device)
            for i in range(self.blocksize):
                w = self.x[:, :, i] * self.x[:, :, i]
                slx = sumlx - w * self.x[:, :, i] * L[:, :, i]
                sl2 = suml2 - w * L[:, :, i] * L[:, :, i]
                new_l = torch.round(self.x[:, :, i] * sl2 / slx)
                new_l = torch.clamp(new_l, -self.nmax, self.nmax - 1)

                slx += w * self.x[:, :, i] * new_l
                sl2 += w * new_l * new_l
                # mask = True means update will happen
                mask_slx = slx > 0  # TODO: check if correct
                mask_new_l = new_l != L[:, :, i]
                mask_better = slx * slx * suml2 > sumlx * sumlx * sl2
                mask = (mask_slx & mask_new_l & mask_better & still_changeable)
                L[:, :, i][mask] = new_l[mask]
                sumlx[mask] = slx[mask]
                suml2[mask] = sl2[mask]
                n_changed[mask] += 1

            mask_no_update = n_changed == 0
            still_changeable[mask_no_update] = False

        for i in range(self.blocksize):
            L[:, :, i] += self.nmax

        scale = sumlx / suml2

        scale[mask_zero_scale] = 0
        L[mask_zero_scale] = 0

        self.L = L
        self.scale = scale

    def _double_quant(self):
        # amax = torch.max(self.scale.abs(), dim=-1).values
        argabsmax_scales = torch.argmax(self.scale.abs(), dim=-1)
        max_scale = torch.gather(self.scale, dim=-1, index=argabsmax_scales.unsqueeze(-1)).squeeze(-1)
        mask_zero_scale = max_scale == 0
        # assert torch.all(max_scale != 0), f"{(max_scale == 0).sum()} {(max_scale == 0).sum() / max_scale.numel() * 100:.2f}%"
        lscales = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
        iscale = -32.0 / max_scale
        for i in range(self.num_blocks):
            l = torch.round(iscale * self.scale[:, i])
            l = torch.clamp(l, -32, 31) + 32
            lscales[:, i] = l
        self.argabsmax_scales = argabsmax_scales
        self.lscales = lscales
        self.d = (1 / iscale).half()

        self.d[mask_zero_scale] = 0
        self.lscales[mask_zero_scale] = 0

    def _final_quant(self):
        for i in range(self.num_blocks):
            sc = self.lscales[:, i] - 32
            d = self.d.float() * sc
            mask = d != 0
            d_nonzero = torch.where(mask, d, torch.ones_like(d))
            for j in range(self.blocksize):
                l = torch.round(self.x[:, i, j] / d_nonzero)
                l = torch.clamp(l, -4, 3)
                self.L[:, i, j][mask] = l[mask] + 4


class Q6KEmulator(GGUFEmulator):
    """explanation"""
    ThisData = Q6KData
    def __init__(self, x: torch.Tensor, **kwargs):
        self.num_bit = 6
        super().__init__(x, **kwargs)

        assert self.num_blocks == 16
        assert self.blocksize == 16

    def _register_params(self, kwargs):
        self.nmax = torch.tensor(kwargs.get("nmax", 32)).to(self.device)  # necessary.

    def quantize(self) -> ThisData:
        with torch.no_grad():
            self._make_qx_quants()
            self._double_quant()
            self._final_quant()
        assert torch.all(self.L >= 0) & torch.all(self.L <= 63), f"{self.L.min()}, {self.L.max()}"
        assert torch.all(self.lscales >= -128) & torch.all(self.lscales <= 127), f"{self.lscales.min()}, {self.lscales.max()}"
        quant_data = Q6KData(
            scale_factors=self.d.detach().cpu().float().numpy().reshape(self.num_superblocks, 1, 1),
            quantized_factors=self.lscales.detach().cpu().to(torch.int8).numpy().reshape(self.num_superblocks, self.num_blocks, 1),
            qs=self.L.detach().cpu().to(torch.int8).numpy() - 32,
        )
        self.quant_data = quant_data
        return quant_data

    def _make_qx_quants(self):
        absmax_val, absmax_idx = self.x.abs().max(dim=-1)
        max_val = self.x.gather(dim=-1, index=absmax_idx.unsqueeze(-1)).squeeze(-1)

        # can be zero (e.g., for untrained embeddings)
        mask_max_zero = max_val == 0  # can't use GROUP_MAX_EPS

        iscale = -self.nmax / max_val

        sumlx = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
        suml2 = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
        L = torch.zeros(self.num_superblocks, self.num_blocks, self.blocksize).to(self.device)

        # First regression solving
        for i in range(self.blocksize):
            l = torch.round(iscale * self.x[:, :, i])
            l = torch.clamp(l, -self.nmax, self.nmax - 1)
            L[:, :, i] = l + self.nmax
            w = self.x[:, :, i] * self.x[:, :, i]
            sumlx += w * self.x[:, :, i] * l
            suml2 += w * l * l

        scale = sumlx / suml2

        scale[mask_max_zero] = 0
        sumlx[mask_max_zero] = 0
        best = scale * sumlx

        # Grid search for a better regression setting
        selected_is = torch.full((self.num_superblocks, self.num_blocks), -100, dtype=torch.int).to(self.device)
        for is_ in range(-9, 10):
            if is_ == 0:
                continue

            iscale = -(self.nmax + 0.1 * is_) / max_val
            sumlx = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
            suml2 = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)

            for i in range(self.blocksize):
                l = torch.round(iscale * self.x[:, :, i])
                l = torch.clamp(l, -self.nmax, self.nmax - 1)
                w = self.x[:, :, i] * self.x[:, :, i]
                sumlx += w * self.x[:, :, i] * l
                suml2 += w * l * l

            need_update = (suml2 > 0) &  (sumlx * sumlx > best * suml2)
            selected_is[need_update] = is_ # for debug

            for i in range(self.blocksize):
                l = torch.round(iscale * self.x[:, :, i])
                L[:, :, i][need_update] = self.nmax + torch.clamp(l[need_update], -self.nmax, self.nmax - 1)
            scale[need_update] = sumlx[need_update] / suml2[need_update]
            best[need_update] = scale[need_update] * sumlx[need_update]

        L[mask_max_zero] = 0
        scale[mask_max_zero] = 0

        assert torch.all(L >= 0) and torch.all(L < 2 * self.nmax), f"[{L.min(), L.max()}] not in [0, {2*self.nmax})"

        self.L = L
        self.scale = scale
        self.selected_is = selected_is # for debug

    def _double_quant(self):
        assert hasattr(self, "scale")
        absmax_scale, argabsmax_scale = self.scale.abs().max(dim=-1)
        max_scale = self.scale.gather(dim=-1, index=argabsmax_scale.unsqueeze(-1)).squeeze(-1)

        mask_zero_scale = max_scale == 0

        iscale = torch.tensor(-128.0).to(self.device) / max_scale
        d = (1 / iscale).half()
        lscales = torch.zeros(self.num_superblocks, self.num_blocks).to(self.device)
        for i in range(self.num_blocks):
            lscales[:, i] = torch.clamp(torch.round(iscale * self.scale[:, i]), -128, 127)

        d[mask_zero_scale] = 0
        lscales[mask_zero_scale] = 0

        assert torch.all(lscales <= 127) and torch.all(lscales >= -128), f"[{lscales.min(), lscales.max()}] not in [-128, 127]"

        self.d = d
        self.argabsmax_scales = argabsmax_scale
        self.lscales = lscales

    def _final_quant(self):
        assert hasattr(self, "lscales")
        assert hasattr(self, "d")

        for j in range(self.num_blocks):
            d = self.d.float() * self.lscales[:, j]
            mask = d != 0
            d_nonzero = torch.where(d != 0, d, torch.ones_like(d))
            for i in range(self.blocksize):
                l = torch.round(self.x[:, j, i] / d_nonzero)
                l = torch.clamp(l, -32, 31)
                self.L[:, j, i][mask] = l[mask] + 32
