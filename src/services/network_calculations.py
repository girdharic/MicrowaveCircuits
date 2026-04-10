"""
network_calculations.py
-----------------------
Conversion utilities between network parameter representations.

Public API
----------
A_to_S(A, B, C, D, Rs, RL) -> dict
    Convert ABCD matrices (possibly block / spectral matrices) to
    S-parameters with port reference impedances Rs (source) and RL (load).
"""

import numpy as np


def A_to_S(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    Rs: float = 1.0,
    RL: float = 1.0,
) -> dict:
    """Convert ABCD matrices to S-parameters.

    Convention used::

        [V1]   [A  B] [ V2]
        [I1] = [C  D] [-I2]

    where *I2* flows **out** of port 2 into the load.

    For scalar (single-frequency) networks the inputs are scalars; for
    spectral / harmonic-balance networks they are square matrices of shape
    ``(Nh, Nh)``.

    The S-parameter expressions (Pozar, "Microwave Engineering", ch. 4) for
    reference impedances *Z1 = Rs* and *Z2 = RL* are::

        denom = A * RL + B + C * Rs * RL + D * Rs

        S11 = inv(denom) @ (A * RL + B - C * Rs * RL - D * Rs)
        S21 = 2 * sqrt(Rs * RL) * inv(denom)
        S12 = 2 * sqrt(Rs * RL) * inv(denom) @ (A @ D - B @ C)
        S22 = inv(denom) @ (-A * Rs + B - C * Rs * RL + D * RL)

    For ``Rs = RL = 1`` these reduce to the familiar textbook forms::

        denom = A + B + C + D
        S11 = inv(denom) @ (A + B - C - D)
        S21 = 2 * inv(denom)
        S12 = 2 * inv(denom) @ (A @ D - B @ C)
        S22 = inv(denom) @ (-A + B - C + D)

    Parameters
    ----------
    A, B, C, D : ndarray
        ABCD sub-matrices (shape ``(Nh, Nh)`` or scalars for single-mode).
    Rs : float
        Source (port-1) reference impedance  [Ω].  Default ``1.0``.
    RL : float
        Load (port-2) reference impedance  [Ω].  Default ``1.0``.

    Returns
    -------
    dict with keys ``'S11'``, ``'S12'``, ``'S21'``, ``'S22'``
        Each value has the same shape as the input matrices.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the denominator matrix is singular at a particular frequency.
    """
    denom = A * RL + B + C * (Rs * RL) + D * Rs
    denom_inv = np.linalg.inv(denom)

    sqrt_ZZ = np.sqrt(Rs * RL)

    S11 = denom_inv @ (A * RL + B - C * (Rs * RL) - D * Rs)
    S21 = (2.0 * sqrt_ZZ) * denom_inv
    S12 = (2.0 * sqrt_ZZ) * denom_inv @ (A @ D - B @ C)
    S22 = denom_inv @ (-A * Rs + B - C * (Rs * RL) + D * RL)

    return {"S11": S11, "S12": S12, "S21": S21, "S22": S22}
