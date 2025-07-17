import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.fft as fft

from neuralop.models import FNO, TFNO  # type: ignore
from neuralop.data.datasets.navier_stokes import NavierStokesDataset

# --------------------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------------------

def psi_from_vorticity(w: torch.Tensor) -> torch.Tensor:
    """Solve Poisson eq ∇²ψ = -ω on a periodic domain using FFT.

    Parameters
    ----------
    w : torch.Tensor, shape (..., H, W)
        Vorticity field.

    Returns
    -------
    ψ : torch.Tensor, same shape as w
    """
    H, W = w.shape[-2:]
    kx = torch.fft.fftfreq(W, d=1.0, device=w.device) * 2 * torch.pi
    ky = torch.fft.fftfreq(H, d=1.0, device=w.device) * 2 * torch.pi
    kx, ky = torch.meshgrid(ky, kx, indexing="ij")
    k_squared = kx ** 2 + ky ** 2
    # Fourier transform of vorticity
    w_hat = fft.fftn(w, dim=(-2, -1))
    # Avoid division by zero at k=0
    k_squared[0, 0] = 1.0
    psi_hat = -w_hat / k_squared
    psi_hat[..., 0, 0] = 0.0  # enforce zero-mean stream function
    psi = fft.ifftn(psi_hat, dim=(-2, -1)).real
    return psi

def velocity_from_psi(psi: torch.Tensor) -> torch.Tensor:
    """Compute velocity field (u_x, u_y) from stream function ψ.

    u_x = ∂ψ/∂y,  u_y = -∂ψ/∂x
    """
    H, W = psi.shape[-2:]
    kx = torch.fft.fftfreq(W, d=1.0, device=psi.device) * 2 * torch.pi
    ky = torch.fft.fftfreq(H, d=1.0, device=psi.device) * 2 * torch.pi
    kx, ky = torch.meshgrid(ky, kx, indexing="ij")
    psi_hat = fft.fftn(psi, dim=(-2, -1))
    ux_hat = 1j * ky * psi_hat
    uy_hat = -1j * kx * psi_hat
    ux = fft.ifftn(ux_hat, dim=(-2, -1)).real
    uy = fft.ifftn(uy_hat, dim=(-2, -1)).real
    return torch.stack([ux, uy], dim=-3)  # shape (2, H, W)

def vorticity_from_velocity(u: torch.Tensor) -> torch.Tensor:
    """Compute vorticity ω = ∂u_y/∂x - ∂u_x/∂y via spectral derivatives.

    u : (2, H, W)
    Returns ω : (H, W)
    """
    ux, uy = u[0], u[1]
    H, W = ux.shape[-2:]
    kx = torch.fft.fftfreq(W, d=1.0, device=ux.device) * 2 * torch.pi
    ky = torch.fft.fftfreq(H, d=1.0, device=ux.device) * 2 * torch.pi
    kx, ky = torch.meshgrid(ky, kx, indexing="ij")
    ux_hat = fft.fftn(ux, dim=(-2, -1))
    uy_hat = fft.fftn(uy, dim=(-2, -1))
    d_uy_dx = fft.ifftn(1j * kx * uy_hat, dim=(-2, -1)).real
    d_ux_dy = fft.ifftn(1j * ky * ux_hat, dim=(-2, -1)).real
    return d_uy_dx - d_ux_dy

# --------------------------------------------------------------------------------------------------
# LBM implementation (D2Q9, BGK)
# --------------------------------------------------------------------------------------------------

C = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.int64)
W = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=torch.float32)
cs2 = 1.0 / 3.0

def equilibrium(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute equilibrium distribution feq.

    rho: (H, W)
    u  : (2, H, W)
    Returns feq: (H, W, 9)
    """
    cu = u.permute(1, 2, 0) @ C.T.float()  # (H,W,9)
    u2 = (u[0] ** 2 + u[1] ** 2)[..., None]
    feq = rho[..., None] * W * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * u2)
    return feq

def lbm_step(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Perform one LBM collision + streaming step on distribution f.

    f: (H, W, 9)
    Returns updated f.
    """
    rho = f.sum(dim=-1)
    # extract velocity
    u = (f @ C.float()) / rho[..., None]
    u = u.permute(2, 0, 1)  # (2,H,W)
    feq = equilibrium(rho, u)
    f_post = f - (f - feq) / tau
    # streaming
    for i, (cx, cy) in enumerate(C):
        f_post[..., i] = torch.roll(f_post[..., i], shifts=(int(cy.item()), int(cx.item())), dims=(0, 1))
    return f_post

# --------------------------------------------------------------------------------------------------
# Hybrid simulation routine
# --------------------------------------------------------------------------------------------------

def hybrid_simulation(model: torch.nn.Module, w0: torch.Tensor, n_steps: int = 50, tau: float = 0.6,
                       device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Run hybrid LBM–FNO simulation starting from vorticity field w0.

    Returns history of vorticity fields, shape (n_steps+1, H, W)
    """
    w = w0.to(device)
    psi = psi_from_vorticity(w)
    u = velocity_from_psi(psi)
    rho = torch.ones_like(w)
    f = equilibrium(rho, u).to(device)

    history = [w.detach().cpu()]

    for step in range(n_steps):
        if step % 2 == 0:  # LBM step
            f = lbm_step(f, tau=tau)
            rho = f.sum(dim=-1)
            u = (f @ C.float()).permute(2, 0, 1) / rho
            w = vorticity_from_velocity(u)
        else:  # FNO prediction step
            inp = w.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            with torch.no_grad():
                w_pred = model(inp).squeeze(0).squeeze(0)
            w = w_pred
            # rebuild velocity & distribution from FNO output
            psi = psi_from_vorticity(w)
            u = velocity_from_psi(psi)
            rho = torch.ones_like(w)
            f = equilibrium(rho, u)
        history.append(w.detach().cpu())
    return torch.stack(history)

# --------------------------------------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hybrid LBM–FNO solver for 2D Navier–Stokes vorticity.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to trained FNO checkpoint (state_dict).")
    parser.add_argument("--data", type=Path, required=True, help="Path to nsforcing_test_128.pt (or similar).")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of sample in dataset to run.")
    parser.add_argument("--n-steps", type=int, default=50, help="Number of hybrid steps to run.")
    parser.add_argument("--tau", type=float, default=0.6, help="LBM relaxation time.")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the vorticity rollout .pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Quick model recreation; assumes FNO architecture used during training.
    # Adjust n_modes/hidden_channels if different.
    model = FNO(n_modes=(16, 16), hidden_channels=64, in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval().to(device)

    data = torch.load(args.data, map_location="cpu")
    w0 = data["x"][args.sample_index]
    if w0.dim() == 2:
        pass
    elif w0.dim() == 3:
        w0 = w0.squeeze(0)
    else:
        raise RuntimeError("Unexpected data shape for vorticity field.")

    history = hybrid_simulation(model, w0, n_steps=args.n_steps, tau=args.tau, device=device)

    if args.save is not None:
        torch.save({"vorticity": history}, args.save)
        print(f"Rollout saved to {args.save}")

    # Print simple diagnostics
    print("Hybrid simulation finished. shape:", history.shape)

if __name__ == "__main__":
    main()