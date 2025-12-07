"""Module for AF3_ReD."""

import jax
import jax.numpy as jnp
import numpy as np
from alphafold3.model import features
from alphafold3.model import model

def pad_positions(positions: jnp.ndarray, max_len: int) -> jnp.ndarray:
    N = positions.shape[0]
    pad_len = max_len - N

    padded = jnp.pad(
        positions,
        ((0, pad_len), (0, 0), (0, 0), (0, 0)),
        mode="constant"
    )

    return padded

def create_bias_positions(example: features.BatchDict, bias_positions_list: list, max_structures: int) -> jnp.ndarray:
    if bias_positions_list:
      bias_positions = jnp.concatenate(bias_positions_list, axis=0)
    else:
      mask_shape = example['pred_dense_atom_mask'].shape
      positions_shape = (mask_shape[0], mask_shape[1], 3)
      bias_positions = jnp.zeros((0, *positions_shape))
      plddt_mask = jnp.ones((mask_shape[0],), dtype=jnp.bool)

    bias_positions_padded = pad_positions(bias_positions, max_structures)

    return bias_positions_padded

def create_fit_mask(example: features.BatchDict, fit_all: bool, fit_chain: list, plddt_mask: jnp.ndarray) -> np.ndarray:
    atoms_layout = example['token_atoms_layout'].item()
    chain_id = atoms_layout.chain_id

    if fit_all:
      fit_mask = np.ones_like(chain_id, dtype=np.bool)
    else:
      fit_mask = np.isin(chain_id, fit_chain)

    fit_mask = fit_mask * plddt_mask[:, None]

    is_protein = example['is_protein']
    num_fit_res = np.sum(fit_mask[:,0] * is_protein)
    assert num_fit_res > 0, "There is no atoms used for RMS fitting. Specify valid chain IDs."

    return fit_mask

def init_plddt_mask(example: features.BatchDict) -> jnp.ndarray:
    mask_shape = example['pred_dense_atom_mask'].shape
    plddt_mask = jnp.ones((mask_shape[0],), dtype=jnp.bool)

    return plddt_mask

def create_plddt_mask(example: features.BatchDict, result: dict, ca_plddt_min: float) -> jnp.ndarray:
    ca_char = np.array([35, 33, 0, 0])
    atom_name_chars = example['ref_atom_name_chars']
    ca_mask = np.all(atom_name_chars == ca_char, axis=-1)

    predicted_lddt = result.get('predicted_lddt')
    ca_plddt = np.sum(predicted_lddt * ca_mask[None, :, :], axis=2)
    ca_plddt_mean = np.mean(ca_plddt, axis=0)
    plddt_mask = jnp.array(ca_plddt_mean > ca_plddt_min)

    return plddt_mask

def update_bias_positions_and_mask(
    bias_positions_list: list,
    bias_mask: jnp.ndarray,
    result: dict,
    num_samples: int,
    mask_position: int
    ) -> tuple[list, jnp.ndarray]:

    bias_positions_list.append(result['diffusion_samples']['atom_positions'])
    next_bias_mask = result['diffusion_samples']['bias_mask']
    bias_mask = bias_mask.at[num_samples * mask_position : num_samples * mask_position + num_samples].set(next_bias_mask)

    return bias_positions_list, bias_mask

def kabsch_align(
    positions: jnp.ndarray,       # (N_res, 24, 3)
    bias_positions: jnp.ndarray,  # (N_bias, N_res, 24, 3)
    valid_mask: jnp.ndarray,      # (N_bias, N_res, 24, 1)
    num_ca,
    ) -> jnp.ndarray:
    """
    Perform Kabsch alignment of multiple bias_positions to a current positions.

    Args:
      positions: current positions, shape (N_res, 24, 3)
      bias_positions: previously sampled positions to align, shape (N_bias, N_res, 24, 3)
      valid_mask: atom mask used for alignment, shape (N_bias, N_res, 24, 1)
      num_ca: the number of C-alpha atoms

    Returns:
      bias_fitted: aligned bias_positions, shape (N_bias, N_res, 24, 3)
    """

    # Centers
    pos_center = jnp.sum(positions * valid_mask[0], axis=(0,1)) / num_ca
    bias_center = jnp.sum(bias_positions * valid_mask, axis=(1,2)) / num_ca
    bias_center = bias_center[:, None, None, :]      # (N_bias, 1, 1, 3)

    # Centered coordinates
    pos_centered = (positions - pos_center) * valid_mask[0]   # (N_res, 24, 3)
    bias_centered = (bias_positions - bias_center) * valid_mask    # (N_bias, N_res, 24, 3)

    # Covariance matrices (N_bias, 3, 3)
    cov = jnp.einsum("ijm,lijk->lmk", pos_centered, bias_centered)

    # SVD and rotation
    u, _, vh = jnp.linalg.svd(cov, full_matrices=False)
    rot = jnp.matmul(vh.transpose(0,2,1), u.transpose(0,2,1))  # (N_bias, 3, 3)

    # Apply rotation and translation (N_bias, N_res, 24, 3)
    return jnp.einsum("lijk,lkm->lijm", bias_centered, rot) + pos_center[None, None, None, :]

def smooth_diffs(diffs: jnp.ndarray, chain_id: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Average diffs using n residues on each side plus the residue itself (a total of 2n + 1 residues).

    Args:
      diffs: C-alpha diffs, shape (N_bias, N_res, 3)
      chain_id: chain ID for each residue, shape (N_res,)
      n: number of neighbors on each side

    Returns:
      smoothed: (N_bias, N_res, 3)
    """
    N_bias, N_res, C = diffs.shape

    smoothed = jnp.zeros_like(diffs)
    counts = jnp.zeros((N_res,))

    # Collect neighbor contributions
    for offset in range(-n, n+1):

        k = abs(offset)
        if offset > 0:
            chain_id_neighbor = jnp.concatenate([jnp.zeros(k, dtype=chain_id.dtype), chain_id[:-k]])
            pad = jnp.zeros((N_bias, k, C), dtype=diffs.dtype)
            diffs_neighbor = jnp.concatenate([pad, diffs[:, :-k, :]], axis=1)
        elif offset < 0:
            chain_id_neighbor = jnp.concatenate([chain_id[k:], jnp.zeros(k, dtype=chain_id.dtype)])
            pad = jnp.zeros((N_bias, k, C), dtype=diffs.dtype)
            diffs_neighbor = jnp.concatenate([diffs[:, k:, :], pad], axis=1)
        else:
            chain_id_neighbor =  chain_id
            diffs_neighbor = diffs

        valid = (chain_id == chain_id_neighbor) & (chain_id != 0)

        smoothed += diffs_neighbor * valid[None, :, None] # (N_bias, N_res, 3)
        counts += valid # (N_res,)

    return smoothed / jnp.maximum(counts[None, :, None], 1.0)

def broadcast_mask(
    mask: jnp.ndarray,            # (N_res, 24), bool
    bias_mask: jnp.ndarray,       # (N_bias,), bool
) -> jnp.ndarray:

    mask_b = mask[None, :, :, None]                 # (1, N_res, 24, 1)
    valid = mask_b * bias_mask[:, None, None, None] # (N_bias, N_res, 24, 1)

    return valid

def bias_grad(
    positions: jnp.ndarray,       # (N_res, 24, 3)
    bias_positions: jnp.ndarray,  # (N_bias, N_res, 24, 3)
    mask: jnp.ndarray,            # (N_res, 24), bool
    ca_mask: jnp.ndarray,         # (N_res, 24), bool
    bias_mask: jnp.ndarray,       # (N_bias,), bool
    chain_id: jnp.ndarray,        # (N_res,), bool
    noise_level: float,
    n_smooth : int,
    sigma : float,
    weight : float,
    bias_tmin : float,
) -> jnp.ndarray:
    """
    Calculate biasing forces.

    Args:
      positions: current positions, shape (N_res, 24, 3)
      bias_positions: previously sampled positions to align, shape (N_bias, N_res, 24, 3)
      mask: atom mask, shape (N_res, 24)
      ca_mask: C-alpha atom mask, shape (N_res, 24)
      bias_mask: which previously sampled positions are valid, shape (N_bias,)
      chain_id: chain ID for each residue (N_res,)
      noise_level: current noise level (= time)
      n_smooth: the number of residues on each side used for diff averaging
      sigma: width of the biasing Gaussian potential
      weight: strength of the biasing Gaussian potential
      bias_tmin: minimum noise level below which no biasing force is applied

    Returns:
      grad: biasing forces, shape (N_res, 24, 3)
    """

    num_ca = jnp.sum(ca_mask)

    # Broadcast masks
    valid = broadcast_mask(mask, bias_mask)
    ca_valid = broadcast_mask(ca_mask, bias_mask)

    # Kabsch alignment
    bias_fitted = kabsch_align(positions, bias_positions, ca_valid, num_ca) # (N_bias, N_res, 24, 3)

    # MSD calculation
    diffs = (positions[None, :, :, :] - bias_fitted) * ca_valid   # (N_bias, N_res, 24, 3)
    msd = jnp.sum(diffs**2, axis=(1,2,3)) / num_ca # (N_bias,)

    # Smoothing
    ca_diffs = jnp.sum(diffs, axis=2) # (N_bias, N_res, 3)
    ca_smoothed = smooth_diffs(ca_diffs, chain_id, n=n_smooth) # (N_bias, N_res, 3)
    ca_smoothed = ca_smoothed[:, :, None, :]              # (N_bias, N_res, 1, 3)
    diffs_smoothed = jnp.tile(ca_smoothed, (1, 1, diffs.shape[2], 1)) * valid  # (N_bias, N_res, 24, 3)

    # Force
    #bias_exp = jnp.exp(- msd / (2.0 * sigma**2))  # (N_bias,)
    #bias_exp = bias_exp[:, None, None, None] # (N_bias, 1, 1, 1)
    #grad = - weight * jnp.sum(bias_exp * diffs_smoothed, axis=0) / (num_ca * sigma**2)  # (N_res, 24, 3) 
    bias_exp = jnp.exp(- msd / (2.0 * (sigma + noise_level)**2))  # (N_bias,)
    bias_exp = bias_exp[:, None, None, None] # (N_bias, 1, 1, 1)
    grad = - weight * jnp.sum(bias_exp * diffs_smoothed, axis=0) / (num_ca * (sigma + noise_level)**2)  # (N_res, 24, 3)

    return grad * (noise_level > bias_tmin)

def get_rmsd(
    positions: jnp.ndarray,       # (N_res, 24, 3)
    bias_positions: jnp.ndarray,  # (N_bias, N_res, 24, 3)
    ca_mask: jnp.ndarray,         # (N_res, 24), bool
    bias_mask: jnp.ndarray,       # (N_bias,), bool
) -> jnp.ndarray:
    """
    Calculare RMSD between multiple bias_positions and a current positions.

    Args:
      positions: current positions, shape (N_res, 24, 3)
      bias_positions: previously sampled positions to align, shape (N_bias, N_res, 24, 3)
      ca_mask: C-alpha atom mask used for alignment, shape (N_res, 24)
      bias_mask: which previously sampled positions are valid, shape (N_bias,)

    Returns:
      rmsd: RMSD values, shape (N_bias,)
    """

    num_ca = jnp.sum(ca_mask)

    # Broadcast masks
    ca_valid = broadcast_mask(ca_mask, bias_mask)

    # Kabsch alignment
    bias_fitted = kabsch_align(positions, bias_positions, ca_valid, num_ca) # (N_bias, N_res, 24, 3)

    # MSD calculation
    diffs = (positions[None, :, :, :] - bias_fitted) * ca_valid   # (N_bias, N_res, 24, 3)
    msd = jnp.sum(diffs**2, axis=(1,2,3)) / num_ca # (N_bias,)

    return jnp.sqrt(msd)
