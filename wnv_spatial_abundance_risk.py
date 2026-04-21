"""
wnv_spatial_abundance_risk.py

End-to-end reference implementation (daily Euler) for:
1) Mosquito abundance (life stages + adult gonotrophic compartments)
2) Spatial movement on a grid via a metapopulation operator R(t)A(t)
   - distance-limited kernel
   - optional "Vic2-style" push–pull: separate emigration pressure (push) and allocation (pull)
3) WNV infection risk layered on top:
   - Mosquito infection SEI within the adult female pool (no double-counting of births/deaths)
   - Bird infection SIR (per-cell or well-mixed)
   - Optional human risk proxy via force of infection

This file is designed to be:
- a readable example for Suman, and for me to put my own rates/functions in later

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Literal

import numpy as np
from scipy import sparse


# ----------------------------
# Utilities
# ----------------------------

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def safe_softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax for a 1D array."""
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 0:
        # fallback uniform
        return np.ones_like(x) / max(1, len(x))
    return ex / s


# ----------------------------
# Parameters
# ----------------------------

@dataclass
class AbundanceParams:
    """
    Parameters for the stage-structured mosquito abundance model.

    NOTE: In the original paper(s), many rates depend on temperature/precip/season.
    Here I provide defaults + hooks so I can implement those functions cleanly later.
    """

    # Egg laying / reproduction
    alpha_A: float = 1.0          # scaling on egg-laying term
    gamma_A0: float = 0.08        # baseline egg production rate (can be time/space dependent)

    # Stage transition rates (can be functions of temperature)
    gamma_L: float = 0.20         # eggs -> larvae
    gamma_P: float = 0.14         # larvae -> pupae
    gamma_A: float = 0.12         # pupae -> adults

    # Adult stage progression rates (gonotrophic cycle)
    gamma_B: float = 0.10         # A -> A_B (blood-fed)
    gamma_En: float = 0.10        # A_B -> A_En (egg-ready "en" stage)
    gamma_El: float = 0.10        # A_En -> A_El (egg-laying)

    # Mortality rates (baseline, per day)
    beta_E: float = 0.05
    beta_L: float = 0.06
    beta_P: float = 0.04
    beta_A: float = 0.03          # adult baseline mortality
    mu_B: float = 0.00            # extra mortality for blood-fed stage

    # Larval density dependence
    beta_1: float = 0.00002       # quadratic density coefficient
    K_L: float = 5000.0           # larval carrying-like scale

    # Pupae emergence penalty
    sigma_A: float = 1.0
    gamma_em: float = 0.2
    K_P: float = 2000.0

    # Diapause switch (0/1) – I'll set per-cell per-day using either the 500 meter set up, or the same previous rule
    psi: float = 1.0

    # Weather-driven larval mortality term placeholder (e.g., washout, dryness)
    beta_W: float = 0.00          # can be made time/space dependent


@dataclass
class ControlParams:
    """
    Vector control knobs.

    We treat larvicide as added larval mortality eta_L,
    and adulticide as added adult mortality zeta (applies to all adult compartments and SEI states).
    """
    eta_L: float = 0.0  # larvicide rate per day (can be cell/day dependent)
    zeta: float = 0.0   # adulticide rate per day (can be cell/day dependent)


@dataclass
class MovementParams:
    """
    Spatial movement parameters (grid metapopulation operator).
    """
    d_max_km: float = 8.0           # max flight radius
    m0: float = 0.10                # baseline mobility scale (per day)
    kernel: Literal["linear"] = "linear"

    # Push–pull options
    use_push_pull: bool = True

    # Emigration pressure coefficients (push)
    a_S: float = 1.5                # spray pushes out
    a_D: float = 0.8                # density pressure pushes out
    a_H: float = 0.6                # poor habitat pushes out
    K_A: float = 5000.0             # scaling for density pressure (adults)

    # Allocation coefficients (pull)
    w_H: float = 1.0                # habitat attractiveness
    w_S: float = 1.0                # spray penalty in destination
    lam: float = 2.0                # softmax sharpness


@dataclass
class InfectionParams:
    """
    Infection dynamics (mosquito SEI + bird SIR).

    IMPORTANT: We do NOT add mosquito births/deaths here (to avoid double-counting).
    Mosquito SEI is updated as flows inside the adult female pool.
    Adulticide/baseline adult mortality is applied via the abundance model.

    Bird model can be per-cell or global (well-mixed).
    """
    # Mosquito infection
    gamma_M: float = 1.0 / 10.0     # exposed->infectious in mosquitoes (EIP), per day (can be temp-dependent)

    # Transmission / biting structure
    # These mirror common FOI forms; feel free to adjust to match your paper's notation.
    alpha_M: float = 0.25           # biting rate scale
    eta: float = 1.0                # seasonal factor (can be time dependent)
    alpha_F: float = 1.0            # feeding preference weighting
    beta_1: float = 0.6             # bird->mosquito transmission probability per bite
    beta_2: float = 0.6             # mosquito->bird transmission probability per bite
    beta_3: float = 0.2             # mosquito->human transmission probability per bite
    phi_B: float = 1.0              # bird susceptibility scaling
    phi_H: float = 1.0              # human susceptibility scaling

    # Bird SIR demography
    b_B: float = 0.0                # bird birth rate per day (can be 0 for seasonal windows)
    m_B: float = 0.0                # bird natural death rate per day
    gamma_B: float = 1.0 / 7.0      # bird recovery rate per day
    delta_B: float = 0.0            # bird WNV mortality per day

    # Human recovery / mortality (optional; for a risk proxy you might not need full SIR)
    gamma_H: float = 1.0 / 10.0
    delta_H: float = 0.0

    # How to handle birds spatially
    birds_mode: Literal["per_cell", "well_mixed"] = "per_cell"


# ----------------------------
# State
# ----------------------------

@dataclass
class State:
    """
    All state variables are vectors of length N (cells), unless otherwise noted.
    """
    # Abundance compartments
    E: np.ndarray
    L: np.ndarray
    P: np.ndarray
    A: np.ndarray
    A_B: np.ndarray
    A_En: np.ndarray
    A_El: np.ndarray

    # Mosquito infection SEI within adult female pool
    S_M: np.ndarray
    E_M: np.ndarray
    I_M: np.ndarray

    # Birds SIR
    S_B: np.ndarray
    I_B: np.ndarray
    R_B: np.ndarray

    # (optional) humans – kept as scalars or vectors if we want
    # Here we store a per-cell "risk proxy" rather than full human compartments.
    risk_lambda_H: np.ndarray


# ----------------------------
# Model
# ----------------------------

class SpatialWNVModel:
    def __init__(
        self,
        cell_xy_km: np.ndarray,  # shape (N,2) in km coordinates
        abundance: AbundanceParams,
        movement: MovementParams,
        infection: InfectionParams,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self.xy = np.asarray(cell_xy_km, dtype=float)
        assert self.xy.ndim == 2 and self.xy.shape[1] == 2
        self.N = self.xy.shape[0]

        self.ab = abundance
        self.mv = movement
        self.inf = infection

        # Precompute distances and neighbor lists for d_max (static geometry)
        self._build_neighbors()

    def _build_neighbors(self):
        N = self.N
        xy = self.xy

        # pairwise distances (N,N) – REMEMBER WESTLEY, it's ok for moderate N; for huge N, use spatial indexing
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        D = np.sqrt(dx * dx + dy * dy)

        self.D = D
        self.neigh: List[np.ndarray] = []
        for j in range(N):
            mask = (D[:, j] > 0) & (D[:, j] < self.mv.d_max_km)
            self.neigh.append(np.where(mask)[0])

        # kernel weights K(d_ij) for neighbors only (stored per origin j)
        self.Kij: List[np.ndarray] = []
        for j in range(N):
            idx = self.neigh[j]
            if len(idx) == 0:
                self.Kij.append(np.zeros((0,), dtype=float))
                continue
            dij = D[idx, j]
            if self.mv.kernel == "linear":
                kij = (self.mv.d_max_km - dij) / self.mv.d_max_km
                kij = np.clip(kij, 0.0, 1.0)
            else:
                raise ValueError(f"Unsupported kernel: {self.mv.kernel}")
            self.Kij.append(kij)

    def init_state(
        self,
        E0=0.0, L0=0.0, P0=0.0, A0=50.0,
        birds_S0=1000.0, birds_I0=1.0, birds_R0=0.0,
    ) -> State:
        N = self.N
        E = np.full(N, E0, dtype=float)
        L = np.full(N, L0, dtype=float)
        P = np.full(N, P0, dtype=float)
        A = np.full(N, A0, dtype=float)

        # Put all adults initially in "A" (not blood-fed etc.)
        A_B = np.zeros(N, dtype=float)
        A_En = np.zeros(N, dtype=float)
        A_El = np.zeros(N, dtype=float)

        # Mosquito infection: start all susceptible, zero exposed/infectious
        # We'll define the "adult female pool" as A + A_B + A_En + A_El
        N_M = A + A_B + A_En + A_El
        S_M = N_M.copy()
        E_M = np.zeros(N, dtype=float)
        I_M = np.zeros(N, dtype=float)

        # Birds
        if self.inf.birds_mode == "well_mixed":
            # store as per-cell constant duplicates (we'll keep them synchronized)
            S_B = np.full(N, birds_S0, dtype=float)
            I_B = np.full(N, birds_I0, dtype=float)
            R_B = np.full(N, birds_R0, dtype=float)
        else:
            S_B = np.full(N, birds_S0, dtype=float)
            I_B = np.full(N, birds_I0, dtype=float)
            R_B = np.full(N, birds_R0, dtype=float)

        risk = np.zeros(N, dtype=float)

        return State(E, L, P, A, A_B, A_En, A_El, S_M, E_M, I_M, S_B, I_B, R_B, risk)

    # ----------------------------
    # External inputs (you will plug your data here)
    # ----------------------------

    def get_controls(self, t_day: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return per-cell control arrays:
        - eta_L[i] larvicide rate (per day)
        - zeta[i] adulticide rate (per day)

        Replace this with your own spray schedule logic.

        Defaults: no control.
        """
        eta_L = np.zeros(self.N, dtype=float)
        zeta = np.zeros(self.N, dtype=float)
        return eta_L, zeta

    def get_habitat_suitability(self, t_day: int) -> np.ndarray:
        """
        Return H_i(t) in [0,1], per cell.
        Replace with weather/land proxies (e.g., temperature suitability, NDVI, water proximity).
        """
        return np.ones(self.N, dtype=float) * 0.5

    def get_spray_intensity_proxy(self, t_day: int) -> np.ndarray:
        """
        Return S_i(t) in [0,1], per cell, representing spray recency/intensity.
        You can build this as an exponential decay after spray nights.
        """
        return np.zeros(self.N, dtype=float)

    # ----------------------------
    # Core computations
    # ----------------------------

    def _build_R(
        self,
        A_adults: np.ndarray,
        H: np.ndarray,
        S: np.ndarray,
    ) -> sparse.csr_matrix:
        """
        Build sparse dispersal matrix R(t) such that move = R * A_adults,
        where off-diagonal r_ij >= 0 for movement j -> i, and diagonal r_jj = -sum_i!=j r_ij.
        """
        N = self.N
        rows = []
        cols = []
        data = []

        # For each origin j, create outflows to neighbors i in N(j)
        for j in range(N):
            nbr = self.neigh[j]
            if len(nbr) == 0:
                # no movement
                continue

            K = self.Kij[j]  # K(d_ij) for i in nbr

            if not self.mv.use_push_pull:
                # Baseline: r_ij = m0 * K(d_ij)
                r_out = self.mv.m0 * K
            else:
                # Vic2-style: emigration pressure m_j(t) then allocation probabilities p_{j->i}(t)

                # Emigration pressure (push)
                # m_j = m0 * exp(a_S*S_j + a_D*(A_j/K_A) + a_H*(1-H_j))
                Aj = max(0.0, float(A_adults[j]))
                mj = self.mv.m0 * np.exp(
                    self.mv.a_S * float(S[j]) +
                    self.mv.a_D * (Aj / max(1e-9, self.mv.K_A)) +
                    self.mv.a_H * (1.0 - float(H[j]))
                )

                # Allocation (pull)
                Ui = self.mv.w_H * H[nbr] - self.mv.w_S * S[nbr]
                logits = self.mv.lam * Ui + np.log(np.maximum(K, 1e-12))  # distance * attractiveness
                p = safe_softmax(logits)  # sums to 1 over neighbors

                r_out = mj * p  # r_ij for i in nbr

            # Populate off-diagonals: row i, col j, value r_ij
            rows.extend(nbr.tolist())
            cols.extend([j] * len(nbr))
            data.extend(r_out.tolist())

            # Diagonal for origin j: r_jj = -sum outflows from j
            rows.append(j)
            cols.append(j)
            data.append(-float(np.sum(r_out)))

        R = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        return R

    def _adult_pool(self, st: State) -> np.ndarray:
        """
        Define which adult compartments are considered the female pool for infection + movement.
        Here: all adult compartments A, A_B, A_En, A_El.
        """
        return st.A + st.A_B + st.A_En + st.A_El

    def step_day(self, st: State, t_day: int, dt: float = 1.0) -> State:
        """
        One day Euler step (dt=1).
        Returns a new State (does not mutate input).
        """
        N = self.N

        # --- External covariates/controls ---
        eta_L, zeta = self.get_controls(t_day)          # per-cell
        H = clamp01(self.get_habitat_suitability(t_day))
        Sspr = clamp01(self.get_spray_intensity_proxy(t_day))

        # --- Copy state ---
        E = st.E.copy()
        L = st.L.copy()
        P = st.P.copy()
        A = st.A.copy()
        A_B = st.A_B.copy()
        A_En = st.A_En.copy()
        A_El = st.A_El.copy()

        # Mosquito infection SEI (inside adult pool)
        S_M = st.S_M.copy()
        E_M = st.E_M.copy()
        I_M = st.I_M.copy()

        # Birds
        S_B = st.S_B.copy()
        I_B = st.I_B.copy()
        R_B = st.R_B.copy()

        # --- Abundance local rates (use hooks to make these weather-dependent) ---
        ab = self.ab

        # Core life-cycle "promotion" flows (unchanged structure)
        # Egg laying depends on "egg-laying adults" A_El
        births_E = ab.gamma_A0 * ab.alpha_A * A_El

        flow_E_to_L = ab.psi * ab.gamma_L * E
        flow_L_to_P = ab.gamma_P * L
        flow_P_to_A = ab.gamma_A * ab.sigma_A * P * np.exp(-ab.gamma_em * (1.0 + (P / max(1e-9, ab.K_P))))

        # Adult gonotrophic flows (A -> A_B -> A_En -> A_El)
        flow_A_to_B = ab.gamma_B * A
        flow_B_to_En = ab.gamma_En * A_B
        flow_En_to_El = ab.gamma_El * A_En
        # Egg-laying stage returns to A via egg-laying “event” implicitly by egg production term;
        # if we want explicit return, add a rate gamma_return * A_El -> A.
        # Here we keep A_El as a compartment that persists until mortality, as in the referenced structure.

        # Deaths + controls
        dE = births_E - flow_E_to_L - ab.beta_E * E
        dL = (flow_E_to_L
              - flow_L_to_P
              - ab.beta_L * L
              - ab.beta_W * L
              - eta_L * L
              - (ab.beta_1 / max(1e-9, ab.K_L)) * (L * L))
        dP = flow_L_to_P - ab.beta_P * P - (ab.gamma_A * P)  # note: gamma_A already used in flow_P_to_A; keep consistent
        # To avoid double-counting, define flow_P_to_A = gamma_A * ... and do dP = flow_L_to_P - beta_P*P - gamma_A*P
        # Here flow_P_to_A already includes gamma_A; so:
        dP = flow_L_to_P - ab.beta_P * P - (ab.gamma_A * P)

        # Adults: local source from pupae emergence; local mortality includes baseline + adulticide zeta.
        # Then add movement operator on the adult pool (applies to all adult compartments).
        adult_mort = (ab.beta_A + zeta)

        # Update adult compartments locally (before movement)
        dA_local = flow_P_to_A - adult_mort * A - flow_A_to_B
        dA_B_local = flow_A_to_B - adult_mort * A_B - ab.mu_B * A_B - flow_B_to_En
        dA_En_local = flow_B_to_En - adult_mort * A_En - flow_En_to_El
        dA_El_local = flow_En_to_El - adult_mort * A_El  # egg-laying captured via births_E

        # --- Movement on adult pool ---
        A_pool = (A + A_B + A_En + A_El)

        # Build R(t) using adult pool and covariates
        Rmat = self._build_R(A_pool, H=H, S=Sspr)
        move_pool = Rmat.dot(A_pool)  # net movement per cell

        # Distribute movement across adult compartments proportionally to their shares in the pool
        # This preserves infection fractions and stage composition under movement.
        with np.errstate(divide="ignore", invalid="ignore"):
            share_A = np.where(A_pool > 0, A / A_pool, 0.0)
            share_AB = np.where(A_pool > 0, A_B / A_pool, 0.0)
            share_AEn = np.where(A_pool > 0, A_En / A_pool, 0.0)
            share_AEl = np.where(A_pool > 0, A_El / A_pool, 0.0)

        dA_move = move_pool * share_A
        dAB_move = move_pool * share_AB
        dAEn_move = move_pool * share_AEn
        dAEl_move = move_pool * share_AEl

        # --- Infection coupling (Bird <-> Mosquito) ---
        # Define per-cell denominators
        N_B = S_B + I_B + R_B
        N_H = np.ones(N, dtype=float)  # if you want real human population per cell, replace this
        denom = (self.inf.alpha_F * N_B + N_H)
        denom = np.maximum(denom, 1e-9)

        # Update mosquito SEI inside adult pool
        # Keep S_M + E_M + I_M aligned to the adult pool after abundance + movement.
        # We'll do:
        # 1) apply movement to S_M/E_M/I_M proportionally to their shares (carry infection with movers)
        # 2) apply infection transitions using FOI
        # 3) renormalize to the current adult pool if small drift occurs

        # 1) move infection compartments with the same movement operator on the adult pool
        N_M = A_pool
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_SM = np.where(N_M > 0, S_M / N_M, 0.0)
            frac_EM = np.where(N_M > 0, E_M / N_M, 0.0)
            frac_IM = np.where(N_M > 0, I_M / N_M, 0.0)

        dSM_move = move_pool * frac_SM
        dEM_move = move_pool * frac_EM
        dIM_move = move_pool * frac_IM

        # Apply movement immediately to SEI (Euler style)
        S_M_m = S_M + dSM_move * dt
        E_M_m = E_M + dEM_move * dt
        I_M_m = I_M + dIM_move * dt

        # 2) Forces of infection (frequency dependent)
        # Mosquitoes get infected from birds:
        lambda_M = (self.inf.alpha_M * self.inf.beta_1 * self.inf.eta * self.inf.alpha_F * I_B) / denom
        # Birds get infected from infectious mosquitoes:
        lambda_B = (self.inf.phi_B * self.inf.alpha_M * self.inf.beta_2 * self.inf.eta * self.inf.alpha_F * I_M_m) / denom
        # Human risk proxy (force of infection from infectious mosquitoes):
        lambda_H = (self.inf.phi_H * self.inf.alpha_M * self.inf.beta_3 * self.inf.eta * I_M_m) / denom

        # SEI flows (no birth/death here; demography handled elsewhere)
        dS_M_inf = -lambda_M * S_M_m
        dE_M_inf = (lambda_M * S_M_m) - (self.inf.gamma_M * E_M_m)
        dI_M_inf = (self.inf.gamma_M * E_M_m)

        S_M_new = S_M_m + dS_M_inf * dt
        E_M_new = E_M_m + dE_M_inf * dt
        I_M_new = I_M_m + dI_M_inf * dt

        # 3) Bird SIR update (can be per-cell or well-mixed)
        dS_B = self.inf.b_B * N_B - self.inf.m_B * S_B - lambda_B * S_B
        dI_B = lambda_B * S_B - self.inf.gamma_B * I_B - self.inf.m_B * I_B - self.inf.delta_B * I_B
        dR_B = self.inf.gamma_B * I_B - self.inf.m_B * R_B

        S_B_new = S_B + dS_B * dt
        I_B_new = I_B + dI_B * dt
        R_B_new = R_B + dR_B * dt

        if self.inf.birds_mode == "well_mixed":
            # Replace with global mean (keeps all cells identical for bird states)
            SBm = float(np.mean(S_B_new))
            IBm = float(np.mean(I_B_new))
            RBm = float(np.mean(R_B_new))
            S_B_new = np.full(N, SBm, dtype=float)
            I_B_new = np.full(N, IBm, dtype=float)
            R_B_new = np.full(N, RBm, dtype=float)

        # --- Apply Euler updates to abundance compartments ---
        E_new = E + dE * dt
        L_new = L + dL * dt
        P_new = P + dP * dt

        A_new = A + (dA_local + dA_move) * dt
        A_B_new = A_B + (dA_B_local + dAB_move) * dt
        A_En_new = A_En + (dA_En_local + dAEn_move) * dt
        A_El_new = A_El + (dA_El_local + dAEl_move) * dt

        # --- Non-negativity clamp (recommended for Euler) ---
        E_new = np.maximum(E_new, 0.0)
        L_new = np.maximum(L_new, 0.0)
        P_new = np.maximum(P_new, 0.0)

        A_new = np.maximum(A_new, 0.0)
        A_B_new = np.maximum(A_B_new, 0.0)
        A_En_new = np.maximum(A_En_new, 0.0)
        A_El_new = np.maximum(A_El_new, 0.0)

        # Recompute adult pool and renormalize SEI to match pool (avoid drift / double counting)
        A_pool_new = A_new + A_B_new + A_En_new + A_El_new
        total_SEI = S_M_new + E_M_new + I_M_new
        # If pool is smaller than total_SEI due to adulticide/mortality, scale down SEI proportionally
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(total_SEI > 0, A_pool_new / total_SEI, 0.0)
        # Cap scale at 1e6 for stability; if A_pool_new > total_SEI, we don't want to inflate infection counts.
        scale = np.clip(scale, 0.0, 1.0)
        S_M_new *= scale
        E_M_new *= scale
        I_M_new *= scale

        S_M_new = np.maximum(S_M_new, 0.0)
        E_M_new = np.maximum(E_M_new, 0.0)
        I_M_new = np.maximum(I_M_new, 0.0)

        # Birds non-negativity
        S_B_new = np.maximum(S_B_new, 0.0)
        I_B_new = np.maximum(I_B_new, 0.0)
        R_B_new = np.maximum(R_B_new, 0.0)

        # Store risk proxy
        risk_new = lambda_H.copy()

        return State(
            E=E_new, L=L_new, P=P_new,
            A=A_new, A_B=A_B_new, A_En=A_En_new, A_El=A_El_new,
            S_M=S_M_new, E_M=E_M_new, I_M=I_M_new,
            S_B=S_B_new, I_B=I_B_new, R_B=R_B_new,
            risk_lambda_H=risk_new
        )

    def run(self, days: int, st0: State) -> Dict[str, np.ndarray]:
        """
        Run model for `days` days. Returns time series arrays:
          - adult_pool[t, i]
          - infectious_mosq[t, i]
          - bird_I[t, i]
          - risk_lambda_H[t, i]
        """
        st = st0
        adult_pool_ts = np.zeros((days + 1, self.N), dtype=float)
        I_m_ts = np.zeros((days + 1, self.N), dtype=float)
        I_b_ts = np.zeros((days + 1, self.N), dtype=float)
        risk_ts = np.zeros((days + 1, self.N), dtype=float)

        adult_pool_ts[0] = self._adult_pool(st)
        I_m_ts[0] = st.I_M
        I_b_ts[0] = st.I_B
        risk_ts[0] = st.risk_lambda_H

        for t in range(days):
            st = self.step_day(st, t_day=t, dt=1.0)
            adult_pool_ts[t + 1] = self._adult_pool(st)
            I_m_ts[t + 1] = st.I_M
            I_b_ts[t + 1] = st.I_B
            risk_ts[t + 1] = st.risk_lambda_H

        return {
            "adult_pool": adult_pool_ts,
            "I_mosquito": I_m_ts,
            "I_bird": I_b_ts,
            "risk_lambda_H": risk_ts,
        }


# ----------------------------
# Example usage (please please please work or I will SCREAM)
# ----------------------------

if __name__ == "__main__":
    # Create a tiny 10x10 grid (100 cells), 1 km spacing
    side = 10
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    xy = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)  # km coordinates

    abundance = AbundanceParams()
    movement = MovementParams(
        d_max_km=3.0,
        m0=0.15,
        use_push_pull=True,
        a_S=1.5, a_D=0.4, a_H=0.4,
        w_H=1.0, w_S=1.0, lam=2.0,
        K_A=2000.0,
    )
    infection = InfectionParams(
        birds_mode="per_cell",
        gamma_M=1.0 / 10.0,
    )

    model = SpatialWNVModel(xy, abundance, movement, infection)

    # Override controls/habitat with simple demos
    def demo_controls(t: int):
        eta_L = np.zeros(model.N)
        zeta = np.zeros(model.N)
        # Spray a 2x2 block around the center for days 20-22
        if 20 <= t <= 22:
            center = (side // 2) * side + (side // 2)
            # pick a few indices
            sprayed = [center, center - 1, center - side, center - side - 1]
            zeta[sprayed] = 0.5  # strong adulticide for demo
        return eta_L, zeta

    def demo_habitat(t: int):
        # Slight gradient habitat
        H = np.linspace(0.2, 0.9, model.N)
        return H

    def demo_spray_proxy(t: int):
        # Recency proxy: 1 on spray day, decays over 5 days
        _, zeta = demo_controls(t)
        S = np.zeros(model.N)
        sprayed = np.where(zeta > 0)[0]
        if len(sprayed) > 0:
            S[sprayed] = 1.0
        return S

    model.get_controls = demo_controls  # type: ignore
    model.get_habitat_suitability = demo_habitat  # type: ignore
    model.get_spray_intensity_proxy = demo_spray_proxy  # type: ignore

    st0 = model.init_state(A0=80.0, birds_S0=800.0, birds_I0=2.0)
    out = model.run(days=60, st0=st0)

    # Print simple diagnostics for review
    print("Adult pool day 0 mean:", out["adult_pool"][0].mean())
    print("Adult pool day 60 mean:", out["adult_pool"][-1].mean())
    print("Infectious mosquito day 60 mean:", out["I_mosquito"][-1].mean())
    print("Risk lambda_H day 60 mean:", out["risk_lambda_H"][-1].mean())
