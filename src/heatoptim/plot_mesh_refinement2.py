#!/usr/bin/env python3
"""
mesh_refinement_plot.py  —  post-processes existing mesh-refinement logs

  ▸ Reads all logs named  sim_res_<res>.log
  ▸ Extracts  (element size  h , average temperature  T_avg, average flux  F_avg)
  ▸ Computes  err_T(h)    = |T_avg(h) − T_avg(h_fine)|
             err_F(h)    = |F_avg(h) − F_avg(h_fine)|
  ▸ Fits  err ≈ C h^p  on the last N_FINE points (default 9) for both T and F
  ▸ Draws four figures:
        (1) err_T   vs h   (log–log, fitted slope p_T shown)
        (2) T_avg   vs h   (semi-log-x)
        (3) err_F   vs h   (log–log, fitted slope p_F shown)
        (4) F_avg   vs h   (semi-log-x)
  ▸ Prints and marks the largest  h  that satisfies  err_T < ΔT_TOL
"""

import glob, os, re, numpy as np, matplotlib.pyplot as plt

# ---------------- USER SETTINGS ---------------------------------------------
LOG_NAME      = "blank"                           # sub-folder
LOG_FOLDER    = os.path.join("mesh_refinement_study", LOG_NAME)
LENGTH        = 5.0e-7                                 # m, domain length scale
DELTA_T_TOL   = 5.0e-5                                 # K, absolute tolerance for T
N_FINE        = 9                                     # points for slope fit
FILE_RX       = re.compile(r"sim_res_([0-9eE+\-.]+)\.log")
TEMP_RX       = re.compile(r"Average\s+temperature\s*:\s*([\d\.\-eE]+)")
FLUX_RX       = re.compile(r"Average\s+flux\s*:\s*([\d\.\-eE]+)")
# ---------------------------------------------------------------------------


def read_logs(folder):
    """
    Scan `folder` for sim_res_*.log
    Return list of (h, T_avg, F_avg) tuples.
    """
    data = []
    for path in glob.glob(os.path.join(folder, "sim_res_*.log")):
        fname = os.path.basename(path)
        m = FILE_RX.match(fname)
        if not m:
            continue
        res = float(m.group(1))                          
        h   = LENGTH / res                               

        T_avg = None
        F_avg = None
        with open(path) as f:
            for line in f:
                if T_avg is None:
                    mT = TEMP_RX.search(line)
                    if mT:
                        T_avg = float(mT.group(1))
                        continue
                if F_avg is None:
                    mF = FLUX_RX.search(line)
                    if mF:
                        F_avg = float(mF.group(1))
                if T_avg is not None and F_avg is not None:
                    break

        if T_avg is None:
            print(f"[Warning] Temperature not found in {fname}")
        if F_avg is None:
            print(f"[Warning] Flux not found in {fname}")
        if T_avg is not None and F_avg is not None:
            data.append((h, T_avg, F_avg))
    return data


def prepare_arrays(data):
    """
    Given list of (h, T_avg, F_avg), sort descending h so index 0 is coarsest,
    -1 is finest. Return arrays h_desc, T_desc, F_desc, err_T, err_F.
    """
    data.sort(key=lambda x: x[0], reverse=True)
    h_desc = np.array([d[0] for d in data])
    T_desc = np.array([d[1] for d in data])
    F_desc = np.array([d[2] for d in data])

    T_ref  = T_desc[-1]    # finest-mesh temperature
    F_ref  = F_desc[-1]    # finest-mesh flux

    err_T  = np.abs(T_desc - T_ref)
    err_F  = np.abs(F_desc - F_ref)

    return h_desc, T_desc, F_desc, err_T, err_F


def fit_slope(h_desc, err_desc, n_last=N_FINE):
    """
    Fit log(err)=logC + p log(h) on the last n_last nonzero points.
    Return (p, C).
    """
    h_fit    = h_desc[-n_last:]
    err_fit  = err_desc[-n_last:]
    mask     = err_fit > 0
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan
    logh = np.log(h_fit[mask])
    loge = np.log(err_fit[mask])
    p, logC = np.polyfit(logh, loge, 1)
    return p, np.exp(logC)


def choose_h_star(h_desc, err_desc, tol):
    """
    Return the largest h whose error < tol, i.e. first index in descending list
    where err_desc < tol.
    """
    idxs = np.where(err_desc < tol)[0]
    if idxs.size == 0:
        return None, None
    i = idxs[0]
    return h_desc[i], err_desc[i]


def plot_all(h, T, err_T, p_T, C_T, 
             F, err_F, p_F, C_F,
             tol_T, h_star_T, err_star_T, 
             outdir):
    os.makedirs(outdir, exist_ok=True)

    # Remove zero-error reference points for plotting
    nz_T = err_T > 0
    h_T_plot, err_T_plot = h[nz_T], err_T[nz_T]

    nz_F = err_F > 0
    h_F_plot, err_F_plot = h[nz_F], err_F[nz_F]

    # ── (1) err_T vs h (log–log) ─────────────────────────────────────────────
    plt.figure(figsize=(6,5))
    plt.loglog(h_T_plot, err_T_plot, 'o-', label="error_T", color='blue')
    if not np.isnan(p_T):
        h_line = np.logspace(np.log10(h.min()), np.log10(h.max()), 100)
        plt.loglog(h_line, C_T * h_line**p_T, '--', label=f"slope_T = {p_T:.2f}", color='black')
    plt.axhline(tol_T, color='red', ls=':', label=f"tol_T = {tol_T:.1e} K")
    if h_star_T is not None:
        plt.plot(h_star_T, err_star_T, 'rd', ms=6, label=f"h*_T = {h_star_T:.2e} m")
        plt.plot([h_star_T, h_star_T], [err_star_T, tol_T], ls=':', color='red')
    plt.xlabel("element size  $h$  (m)")
    plt.ylabel("$|T_{avg}(h) - T_{avg}(h_{min})|$  (K)")
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="upper right")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_T_vs_h.png"))
    plt.close()

    # ── (2) Raw T_avg vs h (semilog-x) ───────────────────────────────────────
    plt.figure(figsize=(6,5))
    plt.semilogx(h, T, 'o-', color='green')
    if h_star_T is not None:
        plt.plot(h_star_T, T[np.where(h == h_star_T)], 'rd')
    plt.xlabel("element size  $h$  (m)")
    plt.ylabel("$T_{avg}$  (K)")
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Tavg_vs_h.png"))
    plt.close()

    # ── (3) err_F vs h (log–log) ─────────────────────────────────────────────
    plt.figure(figsize=(6,5))
    plt.loglog(h_F_plot, err_F_plot, 'o-', label="error_F", color='orange')
    if not np.isnan(p_F):
        h_line = np.logspace(np.log10(h.min()), np.log10(h.max()), 100)
        plt.loglog(h_line, C_F * h_line**p_F, '--', label=f"slope_F = {p_F:.2f}", color='black')
    # No explicit tol_F line unless user defines one
    plt.xlabel("element size  $h$  (m)")
    plt.ylabel("$|F_{avg}(h) - F_{avg}(h_{min})|$  (units)")
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="upper right")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_F_vs_h.png"))
    plt.close()

    # ── (4) Raw F_avg vs h (semilog-x) ────────────────────────────────────────
    plt.figure(figsize=(6,5))
    plt.semilogx(h, F, 'o-', color='purple')
    plt.xlabel("element size  $h$  (m)")
    plt.ylabel("$F_{avg}$  (flux units)")
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Favg_vs_h.png"))
    plt.close()


def main():
    data = read_logs(LOG_FOLDER)
    if not data:
        print("No valid logs in", LOG_FOLDER)
        return

    h, T, F, err_T, err_F = prepare_arrays(data)
    p_T, C_T = fit_slope(h, err_T)
    p_F, C_F = fit_slope(h, err_F)

    h_star_T, err_star_T = choose_h_star(h, err_T, DELTA_T_TOL)

    print(f"⟨grid-convergence⟩  slope_T = {p_T:.3f}")
    if h_star_T is not None:
        print(f"⟨grid-convergence⟩  tol_T = {DELTA_T_TOL:.1e} K first met at "
              f"h*_T = {h_star_T:.2e} m  (err_T = {err_star_T:.2e} K)")
    else:
        print("No mesh meets the prescribed T tolerance.")

    print(f"⟨grid-convergence⟩  slope_F = {p_F:.3f}")

    plot_all(h, T, err_T, p_T, C_T,
             F, err_F, p_F, C_F,
             DELTA_T_TOL, h_star_T, err_star_T,
             LOG_FOLDER)


if __name__ == "__main__":
    main()
