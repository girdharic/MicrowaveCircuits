"""
spectral_network.py
-------------------
Spectral (Floquet / harmonic-balance) building blocks for time-modulated
microwave circuits.

Public API
----------
YLC_spectral(w, wm, m, L, C0, Nhar, phi) -> ndarray
    Admittance matrix of a shunt LC resonator with a sinusoidally modulated
    capacitor, expressed in the harmonic (spectral) domain.

block_chain(blocks) -> (A, B, C, D)
    Cascade a list of ABCD blocks represented as (Nh×Nh) numpy arrays.
"""

import numpy as np


def YLC_spectral(
    w: float,
    wm: float,
    m: float,
    L: float,
    C0: float,
    Nhar: int,
    phi: float,
) -> np.ndarray:
    """Return the spectral admittance matrix of a time-modulated LC resonator.

    The capacitor is modulated as::

        C(t) = C0 * (1 + m * cos(wm * t + phi))

    In the harmonic domain the admittance matrix is tridiagonal of size
    ``Nh = 2*Nhar + 1``.  Row *n* / column *k* represents the coupling from
    the voltage phasor at harmonic *k* to the current phasor at harmonic *n*.

    Parameters
    ----------
    w : float
        Fundamental (carrier) angular frequency  [rad/s].
    wm : float
        Modulation angular frequency  [rad/s].
    m : float
        Modulation index (0 ≤ m < 1).
    L : float
        Resonator inductance  [H].
    C0 : float
        Resonator capacitance (unmodulated)  [F].
    Nhar : int
        Number of sidebands on each side of the carrier.
        The matrix size is ``2*Nhar + 1``.
    phi : float
        Modulation phase  [rad].

    Returns
    -------
    Y : ndarray, shape (2*Nhar+1, 2*Nhar+1), dtype complex
        Spectral admittance matrix of the shunt LC resonator.
    """
    harmonics = np.arange(-Nhar, Nhar + 1)  # shape (Nh,)
    wn = w + harmonics * wm                 # harmonic angular frequencies

    # ── Diagonal: inductor (time-invariant) + capacitor (DC term) ────────
    # Y_L[n,n] = 1 / (j * wn * L)
    # Y_C[n,n] = j * wn * C0
    Y_diag = 1.0 / (1j * wn * L) + 1j * wn * C0

    # ── Sub-diagonal (row n+1, col n): coupling via e^{+j(wm*t+phi)} ─────
    # Current at harmonic n+1 due to voltage at harmonic n:
    #   Y_C[n+1, n] = j * w_{n+1} * C0 * (m/2) * exp(j*phi)
    Y_sub = 1j * wn[1:] * C0 * (m / 2.0) * np.exp(1j * phi)   # length Nh-1

    # ── Super-diagonal (row n-1, col n): coupling via e^{-j(wm*t+phi)} ───
    # Current at harmonic n-1 due to voltage at harmonic n:
    #   Y_C[n-1, n] = j * w_{n-1} * C0 * (m/2) * exp(-j*phi)
    Y_sup = 1j * wn[:-1] * C0 * (m / 2.0) * np.exp(-1j * phi)  # length Nh-1

    Y = np.diag(Y_diag) + np.diag(Y_sub, -1) + np.diag(Y_sup, 1)
    return Y


def block_chain(blocks: list) -> tuple:
    """Cascade a list of ABCD blocks into a single equivalent ABCD matrix.

    Each block is a ``dict`` with keys ``'A'``, ``'B'``, ``'C'``, ``'D'``,
    each holding a square ``numpy.ndarray`` of the same size (Nh × Nh).

    The cascade formula for two consecutive blocks is::

        A12 = A1 @ A2 + B1 @ C2
        B12 = A1 @ B2 + B1 @ D2
        C12 = C1 @ A2 + D1 @ C2
        D12 = C1 @ B2 + D1 @ D2

    Parameters
    ----------
    blocks : list of dict
        Ordered list of ABCD blocks to cascade.

    Returns
    -------
    A, B, C, D : ndarray
        Total ABCD matrices of the cascaded network.

    Raises
    ------
    ValueError
        If ``blocks`` is empty.
    """
    if not blocks:
        raise ValueError("block_chain requires at least one block.")

    A = blocks[0]["A"].copy()
    B = blocks[0]["B"].copy()
    C = blocks[0]["C"].copy()
    D = blocks[0]["D"].copy()

    for blk in blocks[1:]:
        a, b, c, d = blk["A"], blk["B"], blk["C"], blk["D"]
        A_new = A @ a + B @ c
        B_new = A @ b + B @ d
        C_new = C @ a + D @ c
        D_new = C @ b + D @ d
        A, B, C, D = A_new, B_new, C_new, D_new

    return A, B, C, D
