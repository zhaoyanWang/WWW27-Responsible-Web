import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataloaders import SparseDataset,sparse_batch_collate
from . import sd as gen
def loader(matrix, batch_size, seed):
    dataset = SparseDataset(matrix, matrix)
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(
            dataset, generator=torch.Generator(device="cpu").manual_seed(seed)
        ),
        batch_size=batch_size,
        drop_last=False,
    )
    return DataLoader(
        dataset, batch_size=1, collate_fn=sparse_batch_collate,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        sampler=sampler, shuffle=False,
    )

def conditional_decode(matrix, diffusion, vae, timesteps, fraction, noise_scale, seed):
    """Denoise the same user rows from an intermediate diffusion state."""
    diffusion.eval()
    vae.eval()
    vae.is_training = 0
    device = next(vae.parameters()).device
    dense = torch.tensor(matrix.toarray(), dtype=torch.float32, device=device)
    latent, _ = vae.encode(dense)
    start = max(1, min(timesteps, int(round(timesteps * fraction))))
    generator = torch.Generator(device=device).manual_seed(seed + 10_000 + start)
    noise = torch.randn(latent.shape, generator=generator, device=device) * noise_scale
    # Use the same forward perturbation as SD training.
    current = gen.ab_t[start].sqrt() * latent + (1 - gen.ab_t[start]) * noise
    for step in range(start, 0, -1):
        step_tensor = torch.full((len(current),), step, dtype=torch.long, device=device)
        predicted = diffusion(current, step_tensor)
        reverse_noise = (
            torch.randn(current.shape, generator=generator, device=device) * noise_scale
            if step > 1 else torch.zeros_like(current)
        )
        current = gen.denoise_add_noise(current, step, predicted, reverse_noise)
    decoded = vae.decode(current).detach().cpu().numpy().astype(np.float32)
    assert np.isfinite(decoded).all()
    return decoded, start
