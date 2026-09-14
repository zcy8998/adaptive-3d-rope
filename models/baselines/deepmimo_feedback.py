import numpy as np
import torch
from torch import nn

from datasets.deepmimo.csi_grid import (
    batch_target_grid,
    compact_flat_by_mask,
    crop_grid_to_valid_extent,
    flatten_grid,
    scatter_compact_to_flat,
)


def _code_dim(source_dim, compression_ratio, latent_dim=None):
    if latent_dim is not None and int(latent_dim) > 0:
        return int(latent_dim)
    return max(1, int(source_dim) // max(1, int(compression_ratio)))


class _Residual2D(nn.Module):
    def __init__(self, channels=2, hidden=(8, 16)):
        super().__init__()
        hidden_1, hidden_2 = hidden
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_1, 3, padding=1),
            nn.BatchNorm2d(hidden_1),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(hidden_1, hidden_2, 3, padding=1),
            nn.BatchNorm2d(hidden_2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(hidden_2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class _GridFeedbackMixin:
    def _target(self, batch):
        return batch_target_grid(batch)

    def _finish(self, pred_grid, target_grid, valid):
        loss = ((pred_grid - target_grid) ** 2) * valid.to(pred_grid.dtype)
        loss = loss.sum() / valid.sum().clamp_min(1).to(pred_grid.dtype)
        pred_flat, _ = flatten_grid(pred_grid, valid)
        return loss, pred_flat

    def _cropped_target(self, batch):
        target_grid, valid = self._target(batch)
        return crop_grid_to_valid_extent(target_grid, valid, batch)


class CSINetAdapter(nn.Module, _GridFeedbackMixin):
    """CSINet-style convolutional autoencoder adapted to DeepMIMO v4 grids."""

    def __init__(self, compression_ratio=16, latent_dim=None, residual_blocks=2):
        super().__init__()
        self.compression_ratio = int(compression_ratio)
        self.requested_latent_dim = int(latent_dim) if latent_dim else None
        self.residual_blocks = int(residual_blocks)
        self.encoder = None
        self.decoder_in = None
        self.decoder = None
        self.image_shape = None
        self.source_dim = None
        self.latent_dim = None
        self.register_buffer("csinet_scale", torch.tensor(1.0, dtype=torch.float32))

    def _set_scale_from_dataset(self, dataset, device):
        tokens = getattr(dataset, "x", None)
        if tokens is None:
            return
        values = np.concatenate(
            [
                np.asarray(tokens.real, dtype=np.float32).reshape(-1),
                np.asarray(tokens.imag, dtype=np.float32).reshape(-1),
            ]
        )
        scale = float(np.max(np.abs(values))) if values.size else 1.0
        scale = max(scale, 1e-6)
        self.csinet_scale = torch.tensor(scale, dtype=torch.float32, device=device)

    def _to_csinet_domain(self, grid):
        scale = self.csinet_scale.to(device=grid.device, dtype=grid.dtype)
        return (grid / scale + 1.0).mul(0.5).clamp(0.0, 1.0)

    def _from_csinet_domain(self, grid):
        scale = self.csinet_scale.to(device=grid.device, dtype=grid.dtype)
        return grid.mul(2.0).sub(1.0).mul(scale)

    def _build(self, image_shape, source_dim, device):
        _, channels, height, width = image_shape
        image_key = (channels, height, width)
        if self.encoder is not None and self.image_shape == image_key and self.source_dim == source_dim:
            return
        flat_dim = channels * height * width
        self.source_dim = int(source_dim)
        self.latent_dim = _code_dim(source_dim, self.compression_ratio, self.requested_latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 2, 3, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Flatten(),
            nn.Linear(flat_dim, self.latent_dim),
        ).to(device)
        self.decoder_in = nn.Linear(self.latent_dim, flat_dim).to(device)
        self.decoder = nn.Sequential(
            *[_Residual2D(channels) for _ in range(self.residual_blocks)],
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid(),
        ).to(device)
        self.image_shape = image_key

    def build_from_dataset(self, dataset, device):
        dims = dataset.dims[0] if getattr(dataset, "dims", None) is not None else None
        if dims is None:
            return
        self._set_scale_from_dataset(dataset, device)
        orig = dataset.orig_dims[0] if getattr(dataset, "orig_dims", None) is not None else dims
        if torch.is_tensor(orig):
            orig = orig.tolist()
        t, k, u = [int(v) for v in orig]
        image_shape = (1, 2, t * k, u)
        source_dim = int(t * k * u * 2)
        self._build(image_shape, source_dim, device)

    @staticmethod
    def _grid_to_image(grid):
        batch, channels, t, k, u = grid.shape
        return grid.reshape(batch, channels, t * k, u)

    @staticmethod
    def _image_to_grid(image, target_shape):
        batch, channels, t, k, u = target_shape
        return image.reshape(batch, channels, t, k, u)

    def forward(self, batch):
        full_target_grid, full_valid = self._target(batch)
        target_grid, valid = crop_grid_to_valid_extent(full_target_grid, full_valid, batch)
        target_csinet = self._to_csinet_domain(target_grid)
        target_csinet = target_csinet * valid.to(target_csinet.dtype)
        image = self._grid_to_image(target_csinet)
        source_dim = int(valid[0].sum().item())
        self._build(image.shape, source_dim, image.device)
        code = self.encoder(image)
        recon_image = self.decoder(self.decoder_in(code).view_as(image))
        pred_csinet = self._image_to_grid(recon_image, target_grid.shape)
        pred_grid = self._from_csinet_domain(pred_csinet)
        loss = ((pred_csinet - target_csinet) ** 2) * valid.to(pred_csinet.dtype)
        loss = loss.sum() / valid.sum().clamp_min(1).to(pred_csinet.dtype)
        pred_full = torch.zeros_like(full_target_grid)
        pred_full[..., : pred_grid.shape[-3], : pred_grid.shape[-2], : pred_grid.shape[-1]] = pred_grid
        pred_flat, _ = flatten_grid(pred_full, full_valid)
        return loss, pred_flat


class TransNetAdapter(nn.Module, _GridFeedbackMixin):
    """TransNet-style transformer autoencoder adapted to DeepMIMO v4 grids."""

    def __init__(self, compression_ratio=16, latent_dim=None, d_model=64, num_layers=2, nhead=2):
        super().__init__()
        self.compression_ratio = int(compression_ratio)
        self.requested_latent_dim = int(latent_dim) if latent_dim else None
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.nhead = int(nhead)
        self.encoder = None
        self.decoder = None
        self.fc_encoder = None
        self.fc_decoder = None
        self.flat_dim = None
        self.padded_dim = None
        self.source_dim = None
        self.latent_dim = None

    def _build(self, flat_dim, source_dim, device):
        if self.encoder is not None and self.flat_dim == flat_dim and self.source_dim == source_dim:
            return
        self.flat_dim = int(flat_dim)
        self.padded_dim = int(((flat_dim + self.d_model - 1) // self.d_model) * self.d_model)
        self.source_dim = int(source_dim)
        self.latent_dim = _code_dim(source_dim, self.compression_ratio, self.requested_latent_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=max(256, self.d_model * 4),
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )
        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=max(256, self.d_model * 4),
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers).to(device)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=self.num_layers).to(device)
        self.fc_encoder = nn.Linear(self.padded_dim, self.latent_dim).to(device)
        self.fc_decoder = nn.Linear(self.latent_dim, self.padded_dim).to(device)

    def build_from_dataset(self, dataset, device):
        dims = dataset.dims[0] if getattr(dataset, "dims", None) is not None else None
        if dims is None:
            return
        orig = dataset.orig_dims[0] if getattr(dataset, "orig_dims", None) is not None else dims
        if torch.is_tensor(orig):
            orig = orig.tolist()
        source_dim = int(orig[0] * orig[1] * orig[2] * 2)
        self._build(source_dim, source_dim, device)

    def forward(self, batch):
        target_grid, valid = self._target(batch)
        scale = target_grid.abs().flatten(1).amax(dim=1).clamp_min(1e-6)
        normalized_target = target_grid / scale[:, None, None, None, None]
        flat, flat_valid = flatten_grid(normalized_target, valid)
        compact, compact_valid = compact_flat_by_mask(flat, flat_valid)
        source_dim = int(compact_valid[0].sum().item())
        self._build(compact.shape[1], source_dim, compact.device)
        if self.padded_dim > compact.shape[1]:
            flat_in = torch.nn.functional.pad(compact, (0, self.padded_dim - compact.shape[1]))
        else:
            flat_in = compact
        tokens = flat_in.view(flat_in.shape[0], -1, self.d_model)
        memory = self.encoder(tokens)
        code = self.fc_encoder(memory.reshape(memory.shape[0], -1))
        decoded = self.fc_decoder(code).view_as(tokens)
        decoded = self.decoder(decoded).reshape(flat.shape[0], self.padded_dim)
        decoded = decoded[:, : compact.shape[1]]
        required = int(flat_valid.sum(dim=1).max().item())
        if decoded.shape[1] != required:
            raise RuntimeError(
                "TransNet decoder length does not match the valid CSI length: "
                f"decoded={decoded.shape[1]}, required={required}."
            )
        pred_flat = scatter_compact_to_flat(decoded, flat_valid)
        pred_grid = pred_flat.view_as(target_grid)
        pred_grid = pred_grid.tanh() * scale[:, None, None, None, None]
        return self._finish(pred_grid, target_grid, valid)


class CsiNetLSTMAdapter(nn.Module, _GridFeedbackMixin):
    """CsiNet-LSTM-style small recurrent CSI feedback baseline."""

    def __init__(self, compression_ratio=16, latent_dim=None, lstm_layers=2):
        super().__init__()
        self.compression_ratio = int(compression_ratio)
        self.requested_latent_dim = int(latent_dim) if latent_dim else None
        self.lstm_layers = int(lstm_layers)
        self.encoder_conv = None
        self.lstm = None
        self.fc_code = None
        self.fc_decode = None
        self.grid_shape = None
        self.source_dim = None
        self.latent_dim = None

    def _build(self, grid_shape, source_dim, device):
        _, channels, t, k, u = grid_shape
        grid_key = (channels, t, k, u)
        if self.encoder_conv is not None and self.grid_shape == grid_key and self.source_dim == source_dim:
            return
        per_step_dim = channels * k * u
        hidden = min(512, max(64, per_step_dim // 4))
        self.source_dim = int(source_dim)
        self.latent_dim = _code_dim(source_dim, self.compression_ratio, self.requested_latent_dim)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        ).to(device)
        self.lstm = nn.LSTM(
            input_size=per_step_dim,
            hidden_size=hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
        ).to(device)
        self.fc_code = nn.Sequential(nn.Linear(hidden, self.latent_dim), nn.Tanh()).to(device)
        self.fc_decode = nn.Linear(self.latent_dim, channels * t * k * u).to(device)
        self.grid_shape = grid_key

    def build_from_dataset(self, dataset, device):
        dims = dataset.dims[0] if getattr(dataset, "dims", None) is not None else None
        if dims is None:
            return
        orig = dataset.orig_dims[0] if getattr(dataset, "orig_dims", None) is not None else dims
        if torch.is_tensor(orig):
            orig = orig.tolist()
        t, k, u = [int(v) for v in orig]
        grid_shape = (1, 2, t, k, u)
        source_dim = int(orig[0] * orig[1] * orig[2] * 2)
        self._build(grid_shape, source_dim, device)

    def forward(self, batch):
        full_target_grid, full_valid = self._target(batch)
        target_grid, valid = crop_grid_to_valid_extent(full_target_grid, full_valid, batch)
        source_dim = int(valid[0].sum().item())
        self._build(target_grid.shape, source_dim, target_grid.device)
        batch_size, channels, t, k, u = target_grid.shape
        scale = target_grid.abs().flatten(1).amax(dim=1).clamp_min(1e-6)
        normalized_target = target_grid / scale[:, None, None, None, None]
        x = normalized_target.permute(0, 2, 1, 3, 4).reshape(batch_size * t, channels, k, u)
        x = self.encoder_conv(x).reshape(batch_size, t, channels * k * u)
        seq, _ = self.lstm(x)
        code = self.fc_code(seq[:, -1])
        pred_grid = self.fc_decode(code).view_as(target_grid).tanh()
        pred_grid = pred_grid * scale[:, None, None, None, None]
        loss = ((pred_grid - target_grid) ** 2) * valid.to(pred_grid.dtype)
        loss = loss.sum() / valid.sum().clamp_min(1).to(pred_grid.dtype)
        pred_full = torch.zeros_like(full_target_grid)
        pred_full[..., : pred_grid.shape[-3], : pred_grid.shape[-2], : pred_grid.shape[-1]] = pred_grid
        pred_flat, _ = flatten_grid(pred_full, full_valid)
        return loss, pred_flat
