#!/usr/bin/env python3
"""
mesh_refinement_plot.py  —  post-processes existing mesh-refinement logs

  ▸ Reads all logs named  sim_res_<res>.log
  ▸ Extracts  (element size  h , average temperature  T_avg)
  ▸ Computes  err(h) = |T_avg(h) − T_avg(h_fine)|
  ▸ Fits  err ≈ C h^p  on the last N_FINE points (default 3)
  ▸ Draws two figures:
        (1) err  vs  h   (log–log, fitted slope p shown)
        (2) T_avg vs  h   (semi-log-x)
  ▸ Prints and marks the largest  h  that satisfies  err < ΔT_tol
"""

import glob, os, re, numpy as np, matplotlib.pyplot as plt

# ---------------- USER SETTINGS ---------------------------------------------
LOG_NAME = "bimaterial"  # sub-folder
LOG_FOLDER = os.path.join("mesh_refinement_study", LOG_NAME)
LENGTH = 5.0e-7  # m, domain length scale
DELTA_T_TOL = 5.0e-4  # K, absolute tolerance
N_FINE = 9  # points for slope fit
N_PLOT_SKIP = 2  # points to skip in the beginning
FILE_RX = re.compile(r"sim_res_([0-9eE+\-.]+)\.log")
TEMP_RX = re.compile(r"Average\s+temperature\s*:\s*([\d\.eE+\-]+)")
# ---------------------------------------------------------------------------


def read_logs(folder):
    """Return list of (h, T_avg) tuples."""
    pairs = []
    for path in glob.glob(os.path.join(folder, "sim_res_*.log")):
        m = FILE_RX.match(os.path.basename(path))
        if not m:
            continue
        res = float(m.group(1))  # elements per LENGTH
        h = LENGTH / res  # element size
        with open(path) as f:
            for line in f:
                mm = TEMP_RX.search(line)
                if mm:
                    T = float(mm.group(1))
                    pairs.append((h, T))
                    break
    return pairs


def prepare_arrays(pairs):
    """Return sorted arrays  h_desc , T_desc , err_desc ."""
    # sort **descending** so index 0 is coarsest, −1 is finest
    pairs.sort(key=lambda p: p[0], reverse=True)
    # skip the first N_PLOT_SKIP points for plotting
    pairs = pairs[N_PLOT_SKIP:]
    h = np.array([p[0] for p in pairs])
    T = np.array([p[1] for p in pairs])
    T_ref = T[-1]  # finest-mesh temperature
    err = np.abs(T - T_ref)
    return h, T, err


def fit_slope(h_desc, err_desc, n_last=N_FINE):
    """Fit log(err)=logC+p log(h) on the last n_last non-zero points."""
    h_fit = h_desc[-n_last:]
    err_fit = err_desc[-n_last:]
    mask = err_fit > 0
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan
    logh, loge = np.log(h_fit[mask]), np.log(err_fit[mask])
    p, logC = np.polyfit(logh, loge, 1)
    return p, np.exp(logC)


def choose_h_star(h_desc, err_desc, tol):
    """Largest h whose error is already below tolerance."""
    below = np.where(err_desc < tol)[0]
    return (h_desc[below[0]], err_desc[below[0]]) if below.size else (None, None)


def plot_all(h, T, err, p, C, tol, h_star, err_star, outdir):
    os.makedirs(outdir, exist_ok=True)
    # omit the zero-error reference point from the line plot
    nz = err > 0
    h_plot, err_plot = h[nz], err[nz]

    # → Error vs h (log–log)
    plt.figure(figsize=(6, 5))
    plt.loglog(h_plot, err_plot, "o-", label="error", color='blue')
    if not np.isnan(p):
        h_line = np.logspace(np.log10(h.min()), np.log10(h.max()), 100)
        plt.loglog(h_line, C * h_line**p, "--", label=f"slope = {p:.2f}", color='black')
    plt.axhline(tol, color="red", ls=":", label=f"tol = {tol:.1e} K")
    if h_star is not None:
        plt.plot(h_star, err_star, "rd", ms=6, label=f"h* = {h_star:.2e} m")
        # plot vertical line from (h_star, err_star) to (h_star, tol)
        plt.plot([h_star, h_star], [err_star, tol], ls=":", color="red")
    plt.xlabel("element size  $h$  (m)")
    plt.ylabel("$|T_{avg}(h)-T_{avg}(h_{min})|$  (K)")
    # plt.title("Mesh-convergence study")
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="upper right")
    plt.gca().invert_xaxis()  # finer meshes on the right
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_vs_h.png"))
    plt.close()

    # → Raw T_avg vs h
    plt.figure(figsize=(6, 5))
    plt.semilogx(h, T, "o-")
    if h_star is not None:
        plt.plot(h_star, T[np.where(h == h_star)], "rd")
    plt.xlabel("element size  $h$  (m)")
    plt.ylabel("$T_{avg}$  (K)")
    plt.title("$T_{avg}$ vs mesh size")
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Tavg_vs_h.png"))
    plt.close()


def main():
    pairs = read_logs(LOG_FOLDER)
    if not pairs:
        print("No valid logs in", LOG_FOLDER)
        return
    h, T, err = prepare_arrays(pairs)
    p, C = fit_slope(h, err)
    h_star, err_star = choose_h_star(h, err, DELTA_T_TOL)

    print(f"⟨grid-convergence⟩ fitted order  p = {p:.3f}")
    if h_star is not None:
        print(
            f"⟨grid-convergence⟩ tolerance {DELTA_T_TOL:.1e} K first met at "
            f"h* = {h_star:.2e} m  (err = {err_star:.2e} K)"
        )
    else:
        print("No mesh meets the prescribed tolerance.")

    plot_all(h, T, err, p, C, DELTA_T_TOL, h_star, err_star, LOG_FOLDER)


if __name__ == "__main__":
    main()
