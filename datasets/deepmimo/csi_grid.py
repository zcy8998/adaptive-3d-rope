import torch


def tokens_to_complex_grid(tokens, dims, patch_size=4):
    """Invert patch_maker for a batch of complex CSI tokens."""
    if dims.dim() == 1:
        dims = dims.unsqueeze(0).expand(tokens.shape[0], -1)
    dims = dims.to(device=tokens.device)
    max_t = int(dims[:, 0].max().item())
    max_k = int(dims[:, 1].max().item())
    max_u = int(dims[:, 2].max().item())
    grid = torch.zeros(
        tokens.shape[0],
        max_t,
        max_k,
        max_u,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    for idx in range(tokens.shape[0]):
        t, k, u = [int(value.item()) for value in dims[idx]]
        tb, kb, ub = t // patch_size, k // patch_size, u // patch_size
        n_tokens = tb * kb * ub
        sample = tokens[idx, :n_tokens].reshape(
            tb, kb, ub, patch_size, patch_size, patch_size
        )
        sample = sample.permute(0, 3, 1, 4, 2, 5).reshape(t, k, u)
        grid[idx, :t, :k, :u] = sample
    return grid


def complex_grid_to_tokens(grid, patch_size=4):
    """Patchify a padded complex CSI grid using the repo's token ordering."""
    batch, t, k, u = grid.shape
    if t % patch_size or k % patch_size or u % patch_size:
        raise ValueError(
            f"CSI grid shape {(t, k, u)} must be divisible by patch_size={patch_size}."
        )
    tb, kb, ub = t // patch_size, k // patch_size, u // patch_size
    tokens = grid.reshape(
        batch,
        tb,
        patch_size,
        kb,
        patch_size,
        ub,
        patch_size,
    )
    tokens = tokens.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return tokens.reshape(batch, tb * kb * ub, patch_size ** 3)


def complex_grid_to_channels(grid):
    return torch.stack([grid.real, grid.imag], dim=1)


def channels_to_complex_grid(channels):
    return torch.complex(channels[:, 0], channels[:, 1])


def grid_valid_mask(batch, grid=None, include_channels=True):
    orig_dims = batch.get("orig_dims", batch["dims"])
    if orig_dims.dim() == 1:
        orig_dims = orig_dims.unsqueeze(0).expand(batch["x"].shape[0], -1)
    if grid is None:
        dims = batch["dims"]
        if dims.dim() == 1:
            dims = dims.unsqueeze(0).expand(batch["x"].shape[0], -1)
        max_t = int(dims[:, 0].max().item())
        max_k = int(dims[:, 1].max().item())
        max_u = int(dims[:, 2].max().item())
        device = batch["x"].device
    else:
        max_t, max_k, max_u = grid.shape[-3:]
        device = grid.device
    mask = torch.zeros(
        orig_dims.shape[0],
        max_t,
        max_k,
        max_u,
        dtype=torch.bool,
        device=device,
    )
    for idx in range(orig_dims.shape[0]):
        t, k, u = [int(value.item()) for value in orig_dims[idx].to(device="cpu")]
        mask[idx, :t, :k, :u] = True
    if include_channels:
        mask = mask[:, None].expand(-1, 2, -1, -1, -1)
    return mask


def batch_target_grid(batch):
    grid = tokens_to_complex_grid(batch["x"], batch["dims"])
    channels = complex_grid_to_channels(grid).float()
    valid = grid_valid_mask(batch, grid=grid, include_channels=True)
    return channels, valid


def flatten_grid(channels, valid=None):
    flat = channels.reshape(channels.shape[0], -1)
    if valid is None:
        return flat, None
    return flat, valid.reshape(valid.shape[0], -1)


def crop_grid_to_valid_extent(channels, valid, batch):
    """Drop batch-wide padded CSI extent before feeding a baseline codec."""
    orig_dims = batch.get("orig_dims", batch["dims"])
    if orig_dims.dim() == 1:
        orig_dims = orig_dims.unsqueeze(0).expand(channels.shape[0], -1)
    max_t = int(orig_dims[:, 0].max().item())
    max_k = int(orig_dims[:, 1].max().item())
    max_u = int(orig_dims[:, 2].max().item())
    return (
        channels[..., :max_t, :max_k, :max_u],
        valid[..., :max_t, :max_k, :max_u],
    )


def compact_flat_by_mask(flat, mask):
    """Pack true mask positions per sample into a dense [B,max_valid] tensor."""
    mask = mask.to(device=flat.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    max_count = int(counts.max().item()) if counts.numel() else 0
    compact = flat.new_zeros(flat.shape[0], max_count)
    compact_mask = torch.zeros(
        flat.shape[0],
        max_count,
        dtype=torch.bool,
        device=flat.device,
    )
    for idx in range(flat.shape[0]):
        count = int(counts[idx].item())
        if count:
            compact[idx, :count] = flat[idx, mask[idx]]
            compact_mask[idx, :count] = True
    return compact, compact_mask


def scatter_compact_to_flat(compact, mask):
    """Inverse of compact_flat_by_mask for a known original flat mask."""
    mask = mask.to(device=compact.device, dtype=torch.bool)
    flat = compact.new_zeros(mask.shape)
    counts = mask.sum(dim=1)
    for idx in range(mask.shape[0]):
        count = int(counts[idx].item())
        if count:
            if compact.shape[1] < count:
                raise ValueError(
                    "Compact CSI tensor is shorter than its target mask: "
                    f"sample={idx}, compact={compact.shape[1]}, required={count}."
                )
            flat[idx, mask[idx]] = compact[idx, :count]
    return flat


def source_real_dim_from_dims(dims):
    if torch.is_tensor(dims):
        values = dims[0] if dims.dim() == 2 else dims
        return int(values[0].item() * values[1].item() * values[2].item() * 2)
    return int(dims[0] * dims[1] * dims[2] * 2)
