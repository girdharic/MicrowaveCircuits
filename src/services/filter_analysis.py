"""
filter_analysis.py
------------------
Spectral S-parameter analysis of a 3-resonator parametric bandpass filter.

The filter topology is::

    Port1 – J12 – R1 – J23 – R2 – J23 – R3 – J12 – Port2

where:
  * R1, R2, R3  are shunt LC resonators with sinusoidally modulated
                capacitances (phase-shifted between resonators).
  * J12, J23    are ideal J-inverters (admittance inverters).

Public API
----------
Compute_Inverters(BW_MHz, f0_GHz, C0_F, M12, M23) -> (J12, J23)
    Compute the J-inverter coupling values from the normalised coupling-matrix
    coefficients and filter parameters.

compute_S_parameters(f0_GHz, BW_MHz, M12, M23, C0_F, Nhar, fm, m, delta,
                     fGHz=None) -> (S11dB, S22dB, S21dB, S12dB)
    Full spectral S-parameter sweep.
"""

import logging

import numpy as np

from src.services import spectral_network as sn
from src.services import network_calculations as nc

logger = logging.getLogger(__name__)


# ── Helper: J-inverter values ─────────────────────────────────────────────────


def Compute_Inverters(
    BW_MHz: float,
    f0_GHz: float,
    C0_F: float,
    M12: float,
    M23: float,
) -> tuple:
    """Compute J-inverter coupling values for the 3-resonator bandpass filter.

    In the 1-Ω normalised spectral-ABCD framework the J-inverter admittance
    parameter equals the coupling coefficient::

        FBW  = BW_MHz × 10⁶ / (f0_GHz × 10⁹)   [fractional bandwidth]
        J_ij = M_ij_norm × FBW                    [dimensionless in 1-Ω system]

    These values are used directly in the J-inverter ABCD block::

        [  0    1/J ]
        [  J     0  ]

    Parameters
    ----------
    BW_MHz : float
        Filter 3-dB bandwidth  [MHz].
    f0_GHz : float
        Filter centre frequency  [GHz].
    C0_F : float
        Resonator capacitance  [F] (used for future extension; not needed for
        the normalised coupling formula).
    M12 : float
        Normalised coupling-matrix element between resonators 1 and 2.
    M23 : float
        Normalised coupling-matrix element between resonators 2 and 3.

    Returns
    -------
    M25, M56 : float
        J-inverter coupling coefficients (dimensionless in 1-Ω normalised
        system; equivalent to coupling strength in Siemens when Z₀ = 1 Ω).
    """
    FBW = (BW_MHz * 1e6) / (f0_GHz * 1e9)    # fractional bandwidth
    M25 = M12 * FBW
    M56 = M23 * FBW
    return M25, M56


# ── NaN interpolation helper ──────────────────────────────────────────────────


def _fill_nan_interp(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN / ±Inf samples in a complex array."""
    arr = arr.copy()
    xs = np.arange(len(arr))
    for attr in ("real", "imag"):
        vals = getattr(arr, attr).copy()
        bad = ~np.isfinite(vals)
        if bad.any():
            vals[bad] = np.interp(xs[bad], xs[~bad], vals[~bad])
            if attr == "real":
                arr = vals + 1j * arr.imag
            else:
                arr = arr.real + 1j * vals
    return arr


# ── Main function ─────────────────────────────────────────────────────────────


def compute_S_parameters(
    f0_GHz: float,
    BW_MHz: float,
    M12: float,
    M23: float,
    C0_F: float,
    Nhar: int,
    fm: float,
    m: float,
    delta: float,
    fGHz: np.ndarray = None,
) -> tuple:
    """Compute the S-parameters of a 3-resonator parametric bandpass filter.

    The filter uses shunt LC resonators whose capacitances are sinusoidally
    modulated at frequency *fm*, with a progressive phase shift *delta*
    between adjacent resonators.  The spectral (Floquet) ABCD formalism is
    used to capture harmonic mixing products up to order *Nhar*.

    Only the fundamental-to-fundamental (fund → fund) element of each
    S-parameter block matrix is returned.

    Parameters
    ----------
    f0_GHz : float
        Filter centre frequency  [GHz].
    BW_MHz : float
        Filter 3-dB bandwidth  [MHz].
    M12 : float
        Normalised coupling-matrix element between resonators 1 and 2.
    M23 : float
        Normalised coupling-matrix element between resonators 2 and 3.
    C0_F : float
        Resonator capacitance (unmodulated)  [F].
    Nhar : int
        Number of harmonic sidebands retained on each side of the carrier
        (matrix size = 2*Nhar + 1).
    fm : float
        Modulation frequency  [MHz].
    m : float
        Capacitance modulation index (0 ≤ m < 1).
    delta : float
        Progressive phase shift between adjacent resonators  [degrees].
    fGHz : array_like, optional
        Frequency sweep  [GHz].  If *None*, a span of ±5 × BW around *f0*
        is used (1 001 points).

    Returns
    -------
    S11dB : ndarray
        S11 magnitude  [dB] over the frequency sweep.
    S22dB : ndarray
        S22 magnitude  [dB] over the frequency sweep.
    S21dB : ndarray
        S21 magnitude  [dB] over the frequency sweep.
    S12dB : ndarray
        S12 magnitude  [dB] over the frequency sweep.
    """
    # ── Frequency sweep ───────────────────────────────────────────────────
    if fGHz is None:
        f_span_GHz = 5.0 * BW_MHz / 1e3
        fGHz = np.linspace(f0_GHz - f_span_GHz, f0_GHz + f_span_GHz, 1001)
    fGHz = np.asarray(fGHz, dtype=float)

    # ── Constants ─────────────────────────────────────────────────────────
    Rs = 1.0
    RL = 1.0
    wm = 2.0 * np.pi * fm * 1e6              # modulation angular frequency [rad/s]

    harmonics = np.arange(-Nhar, Nhar + 1)
    Nh = len(harmonics)
    fund_idx = Nh // 2                        # index of the 0-th harmonic

    U = np.eye(Nh, dtype=complex)

    # Progressive phase offsets for the three resonators
    phi = np.array([0.0, delta * np.pi / 180.0, 2.0 * delta * np.pi / 180.0])

    # ── Pre-compute frequency-independent quantities ──────────────────────
    w0 = 2.0 * np.pi * f0_GHz * 1e9          # carrier angular frequency [rad/s]
    L = 1.0 / (w0**2 * C0_F)                  # resonator inductance [H]
    logger.debug("C0 = %.3e F, L = %.3e H", C0_F, L)

    M25, M56 = Compute_Inverters(
        BW_MHz=BW_MHz, f0_GHz=f0_GHz, C0_F=C0_F, M12=M12, M23=M23
    )
    logger.debug("J12 (M25) = %.3e, J23 (M56) = %.3e", M25, M56)

    # ── Output arrays ─────────────────────────────────────────────────────
    S11 = np.full(len(fGHz), np.nan, dtype=complex)
    S22 = np.full(len(fGHz), np.nan, dtype=complex)
    S21 = np.full(len(fGHz), np.nan, dtype=complex)
    S12 = np.full(len(fGHz), np.nan, dtype=complex)

    # ── Frequency sweep ───────────────────────────────────────────────────
    for idx, f in enumerate(fGHz):
        w = 2.0 * np.pi * f * 1e9            # angular frequency [rad/s]

        try:
            # Spectral admittance of each shunt resonator
            YLC1 = sn.YLC_spectral(w, wm, m, L, C0_F, Nhar, phi[0])
            YLC2 = sn.YLC_spectral(w, wm, m, L, C0_F, Nhar, phi[1])
            YLC3 = sn.YLC_spectral(w, wm, m, L, C0_F, Nhar, phi[2])

            # J-inverter ABCD blocks: [0, 1/J; J, 0]
            J12 = {"A": 0.0 * U, "B": U / M25, "C": U * M25, "D": 0.0 * U}
            J23 = {"A": 0.0 * U, "B": U / M56, "C": U * M56, "D": 0.0 * U}

            # Shunt-resonator ABCD blocks: [1, 0; Y, 1]
            R1 = {"A": U, "B": 0.0 * U, "C": YLC1, "D": U}
            R2 = {"A": U, "B": 0.0 * U, "C": YLC2, "D": U}
            R3 = {"A": U, "B": 0.0 * U, "C": YLC3, "D": U}

            # Topology: J12 – R1 – J23 – R2 – J23 – R3 – J12
            At, Bt, Ct, Dt = sn.block_chain([J12, R1, J23, R2, J23, R3, J12])

            S = nc.A_to_S(At, Bt, Ct, Dt, Rs, RL)

            # Extract fundamental-to-fundamental element
            S11[idx] = S["S11"][fund_idx, fund_idx]
            S22[idx] = S["S22"][fund_idx, fund_idx]
            S21[idx] = S["S21"][fund_idx, fund_idx]
            S12[idx] = S["S12"][fund_idx, fund_idx]

        except np.linalg.LinAlgError:
            logger.warning(
                "Singular matrix at f=%.6f GHz (idx=%d); point left as NaN.",
                f, idx,
            )  # NaN remains → interpolated below

    # ── Interpolate singular / NaN points ────────────────────────────────
    S11 = _fill_nan_interp(S11)
    S22 = _fill_nan_interp(S22)
    S21 = _fill_nan_interp(S21)
    S12 = _fill_nan_interp(S12)

    # ── Convert to dB ────────────────────────────────────────────────────
    eps = 1e-12
    S11dB = 20.0 * np.log10(np.maximum(np.abs(S11), eps))
    S22dB = 20.0 * np.log10(np.maximum(np.abs(S22), eps))
    S21dB = 20.0 * np.log10(np.maximum(np.abs(S21), eps))
    S12dB = 20.0 * np.log10(np.maximum(np.abs(S12), eps))

    return S11dB, S22dB, S21dB, S12dB
