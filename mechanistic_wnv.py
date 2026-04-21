# mechanistic_wnv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import numpy as np

# ============================================================
# Acta Tropica 2024 (Bhowmick et al.)-style WNV model core
# - Implements eqs. (1)–(12) with sensible defaults.
# - IMPORTANT: To get "real numbers", you MUST calibrate:
#   - bird density NB and human density NH per cell (or proxies)
#   - initial conditions / seeding
#   - temperature functional forms bM(T), mM(T), gammaM(T), eta(T), alphaM(D)
# ============================================================

@dataclass
class WNVParams:
    # Bird demography (Table 2 shows example values)
    bB: float = 0.00342
    mB: float = 0.0012
    gammaB: float = 0.182
    deltaB: float = 0.26

    # Human recovery/mortality (simple SIR; if you want split, use p)
    gammaH: float = 0.5
    deltaH: float = 0.004
    p_neuro: float = 0.006  # fraction neuroinvasive (eq. 8)

    # Transmission probabilities (0..1) - YOU SHOULD SET / FIT THESE
    beta1: float = 0.25  # IB -> mosquito
    beta2: float = 0.25  # IM -> bird
    beta3: float = 0.10  # IM -> human

    # Feeding index for birds (alpha_F in eqs. 4–6)
    alphaF: float = 10.0  # paper says range ~5–40

    # Mosquito-to-host ratios (phi_B, phi_H)
    phiB: float = 15.0
    phiH: float = 0.03

    # Mosquito SEI: incubation rate gammaM(T) (1/day) will be temperature-driven;
    # We keep a fallback base here.
    gammaM_base: float = 1.0 / 10.0  # ~10-day EIP fallback

    # Adulticide pulse (eq. 10)
    zeta0: float = 0.5
    ulv_apply_day: Optional[int] = None        # day-of-year when applied
    ulv_duration_days: int = 7

    # Continuous introduction into birds (eq. 11)
    intro_A0: float = 0.0
    intro_Am: float = 200.0
    intro_Aw: float = 30.0

    # Scaling to map "model mosquito units" to your UI expectations
    # If you want outputs that resemble "mosquitoes per trap-night",
    # you will calibrate this against your trap counts.
    mosq_scale: float = 1.0


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ---------------------------
# Temperature-driven functions
# ---------------------------

def eta_biting_rate(T_c: float) -> float:
    """
    Temperature-dependent biting rate η(T).
    Paper references Laperriere/Bhowmick; exact function is cited externally.
    Here is a safe monotone proxy:
      - near zero below ~10C
      - increases up to ~0.3–0.4/day around 25–30C
    Replace with the exact function from the cited source when you have it.
    """
    T = float(T_c)
    if T < 10.0:
        return 0.01
    # smooth rise then slight plateau
    return float(0.05 + 0.35 * (1.0 / (1.0 + math.exp(-(T - 20.0) / 3.0))))


def alphaM_non_diapausing(daylength_hours: float) -> float:
    """
    Fraction non-diapausing mosquitoes αM(D).
    Paper uses Laperriere et al. function of daylength D (hours).
    Here: a simple sigmoid centered ~12.5h daylength.
    """
    D = float(daylength_hours)
    return clamp01(1.0 / (1.0 + math.exp(-(D - 12.5) / 0.5)))


def mosquito_birth_rate_bM(T_c: float) -> float:
    """
    bM(T): mosquito birth rate in eq. (1).
    Replace with exact function when you have it.
    """
    T = float(T_c)
    if T < 10.0:
        return 0.02
    return float(0.05 + 0.15 * (1.0 / (1.0 + math.exp(-(T - 18.0) / 3.0))))


def mosquito_mortality_mM(T_c: float) -> float:
    """
    mM(T): mosquito mortality in eq. (1).
    Typically U-shaped; here a safe proxy with higher death at extremes.
    Replace with exact function when you have it.
    """
    T = float(T_c)
    # baseline + penalty away from 25C
    return float(0.05 + 0.002 * ((T - 25.0) ** 2) / 25.0)


def mosquito_incubation_gammaM(T_c: float, p: WNVParams) -> float:
    """
    γM(T): exposed -> infected (1/day). Proxy: faster at higher T.
    """
    T = float(T_c)
    if T < 12.0:
        return 1.0 / 20.0
    if T > 30.0:
        return 1.0 / 6.0
    # interpolate 12C->30C from 1/20 to 1/6
    return float((1.0 / 20.0) + (T - 12.0) * ((1.0 / 6.0) - (1.0 / 20.0)) / (30.0 - 12.0))


# ---------------------------
# ULV pulse ζ(t) (eq. 10)
# ---------------------------

def zeta_ulv(day_of_year: int, p: WNVParams) -> float:
    if p.ulv_apply_day is None:
        return 0.0
    d0 = int(p.ulv_apply_day)
    d1 = d0 + int(p.ulv_duration_days)
    return float(p.zeta0) if (d0 <= int(day_of_year) <= d1) else 0.0


# ---------------------------
# Bird introduction ψB(t) (eq. 11)
# ---------------------------

def psiB_introduction(t_day: float, p: WNVParams) -> float:
    """
    ψB(t) = A0 * floor( exp((Am - t)/Aw) / (1+exp((Am - t)/Aw))^2 )
    Text extraction loses floor/symbols. We'll implement the smooth core term.
    """
    A0, Am, Aw = float(p.intro_A0), float(p.intro_Am), float(p.intro_Aw)
    if A0 <= 0.0:
        return 0.0
    x = (Am - float(t_day)) / max(1e-6, Aw)
    ex = math.exp(x)
    return float(A0 * (ex / ((1.0 + ex) ** 2)))


# ---------------------------
# Forces of infection (eqs. 4–6)
# ---------------------------

def forces_of_infection(
    T_c: float,
    alphaM: float,
    eta: float,
    p: WNVParams,
    IB: float,
    NB: float,
    IM: float,
    NH: float
) -> Tuple[float, float, float]:
    """
    λM = αM β1 η αF IB / (αF NB + NH)
    λB = φB αM β2 η αF IM / (αF NB + NH)
    λH = φH αM β3 η IM / (αF NB + NH)
    """
    denom = (p.alphaF * float(NB) + float(NH))
    denom = max(1e-9, denom)

    lamM = alphaM * p.beta1 * eta * p.alphaF * float(IB) / denom
    lamB = p.phiB * alphaM * p.beta2 * eta * p.alphaF * float(IM) / denom
    lamH = p.phiH * alphaM * p.beta3 * eta * float(IM) / denom
    return float(lamM), float(lamB), float(lamH)


# ============================================================
# Main simulator (daily Euler) for one cell
# ============================================================

@dataclass
class WNVState:
    # Mosquito SEI
    SM: float
    EM: float
    IM: float
    # Bird SIR
    SB: float
    IB: float
    RB: float
    # Human SIR (simple)
    SH: float
    IH: float
    RH: float


def simulate_cell_week(
    *,
    # Inputs
    T_c: float,
    day_of_year_start: int,
    daylength_hours: float,
    NB: float,
    NH: float,
    # Params + initial state
    params: WNVParams,
    state0: WNVState,
    # Numerics
    dt: float = 1.0,        # days
    steps: int = 7
) -> Dict[str, float]:
    """
    Simulate eqs. (1),(2),(3) with FOI eqs. (4)-(6),
    optionally with ULV eq. (9)-(10) and bird intro eq. (12).
    """

    s = WNVState(**state0.__dict__)  # copy
    T = float(T_c)

    bM = mosquito_birth_rate_bM(T)
    mM = mosquito_mortality_mM(T)
    gM = mosquito_incubation_gammaM(T, params)
    eta = eta_biting_rate(T)
    aM = alphaM_non_diapausing(daylength_hours)

    for k in range(int(steps)):
        day = int(day_of_year_start) + k
        zeta = zeta_ulv(day, params)
        psiB = psiB_introduction(day, params)

        NM = max(1e-9, s.SM + s.EM + s.IM)
        NBt = max(1e-9, s.SB + s.IB + s.RB)

        lamM, lamB, lamH = forces_of_infection(
            T_c=T, alphaM=aM, eta=eta, p=params,
            IB=s.IB, NB=NBt, IM=s.IM, NH=(s.SH + s.IH + s.RH)
        )

        # -----------------
        # Mosquito SEI (eq. 1) and adulticide (eq. 9)
        # -----------------
        dSM = bM * NM - (mM + zeta) * s.SM - lamM * s.SM
        dEM = lamM * s.SM - gM * s.EM - (mM + zeta) * s.EM
        dIM = gM * s.EM - (mM + zeta) * s.IM

        # -----------------
        # Bird SIR (eq. 2) + intro into IB (eq. 12)
        # -----------------
        dSB = params.bB * NBt - params.mB * s.SB - lamB * s.SB
        dIB = lamB * s.SB - params.gammaB * s.IB - params.mB * s.IB - params.deltaB * s.IB + (s.IB * psiB)
        dRB = params.gammaB * s.IB - params.mB * s.RB

        # -----------------
        # Human SIR (eq. 3) (simple)
        # -----------------
        # If you want split neuroinvasive/non-neuroinvasive, implement eq. (8) instead.
        dSH = -lamH * s.SH
        dIH = lamH * s.SH - params.gammaH * s.IH - params.deltaH * s.IH
        dRH = params.gammaH * s.IH

        # Euler step + positivity
        s.SM = max(0.0, s.SM + dt * dSM)
        s.EM = max(0.0, s.EM + dt * dEM)
        s.IM = max(0.0, s.IM + dt * dIM)

        s.SB = max(0.0, s.SB + dt * dSB)
        s.IB = max(0.0, s.IB + dt * dIB)
        s.RB = max(0.0, s.RB + dt * dRB)

        s.SH = max(0.0, s.SH + dt * dSH)
        s.IH = max(0.0, s.IH + dt * dIH)
        s.RH = max(0.0, s.RH + dt * dRH)

    NM_end = s.SM + s.EM + s.IM
    prevM = (s.IM / max(1e-9, NM_end)) * 100.0

    # Apply a scale factor so you can calibrate "model mosquitoes" to "mosquitoes per trap-night"
    total_scaled = params.mosq_scale * NM_end
    infected_scaled = params.mosq_scale * s.IM

    return {
        "NM_total": float(total_scaled),
        "NM_infected": float(infected_scaled),
        "mosq_prevalence_pct": float(prevM),
        "IB": float(s.IB),
        "NB": float(s.SB + s.IB + s.RB),
        "IH": float(s.IH),
        "NH": float(s.SH + s.IH + s.RH),
    }