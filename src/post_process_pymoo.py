import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np
import json
import os

# sys.path
# sys.path.append('/home/matei/PhD/GAOpt/')


class PymooPostprocess:
    """post processing for pymoo plots
    e.g: pareto front, hypervolume convergence, running metric
    """

    def __init__(self, ITER_PATH):
        with open(ITER_PATH + "/NSGA_Result.pk1", "rb") as f:
            self.res = pickle.load(f)

    def print_termination_reason(self):
        print("Termination percentage", self.res.algorithm.termination.perc)
        print("Termination n_gen", self.res.algorithm.n_gen)

    def get_min_max_pareto(self, nds):
        """for scaling purposes
        parameter: nds, list of non dominated solutions"""
        minF = nds[np.argmin(nds[:, 0])]
        maxF = nds[np.argmax(nds[:, 0])]
        print(f"minF {minF}")
        print(f"maxF {maxF}")
        return minF, maxF

    def get_pareto_plot(
        self,
        path=None,
        plot_compromise=False,
        filter_outliers=False,
        threshold=10000,
        minmax=False,
    ):
        """Saves pareto plot, 2d or 3d depending on problem
        inputs: path-> path to save the plot
                plot_compromise-> plot the compromise solution point
                filter_outliers-> filter out solutions above threshold
                threshold-> threshold value
                minmax-> normalize the pareto front
        """
        F = self.res.F.copy()

        if filter_outliers:
            F = F[np.all(F < threshold, axis=1)]
            from pymoo.decomposition.asf import ASF

            N_OBJ = F.shape[1]
            weights = np.array([0.0001] * N_OBJ)
            decomp = ASF()
            dec_idx = decomp(F, weights).argmin()
        # else:
        #     dec_idx = self.decomp_idx

        if len(F[0]) == 2:
            plt.clf()
            fig = plt.figure(figsize=(10, 8))
            if minmax is True:
                minF, maxF = self.get_min_max_pareto(F)
                F = (F - minF) / (maxF - minF)
                F[:, 1] = 1 - F[:, 1]
            plt.scatter(
                F[:, 0],
                F[:, 1],
                s=30,
                facecolors="none",
                edgecolors="r",
                label="Solutions",
            )
            plt.xlabel("Average temperature")
            plt.ylabel("Temperature std dev")
            # plt.title("Pareto Front")
            plt.savefig(path + "/2D_pareto_front.png")
            plt.close(fig)

        elif len(F[0]) == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], c="r", marker="o")
            ax.set_xlabel("Objective 1: Tsub")
            ax.set_ylabel("Objective 2: Zero Flux")
            ax.set_zlabel("Objective 3: Max Flux")
            # plt.title("Pareto Front")

            if plot_compromise:
                ax.scatter(
                    F[dec_idx, 0],
                    F[dec_idx, 1],
                    F[dec_idx, 2],
                    c="blue",
                    marker="o",
                    s=50,
                    edgecolors="black",
                    linewidths=2,
                    alpha=0.9,
                )
                arrow_start = np.array([0] * 3)
                arrow_end = np.array([0.0001] * 3)
                arrow = arrow_end - arrow_start

                ax.quiver(
                    *arrow_start,
                    *arrow,
                    color="black",
                    alpha=0.4,
                    arrow_length_ratio=0.1,
                )

            # plt.show()
            plt.savefig(
                path + "/3D_pareto_front.png", pad_inches=0.5, bbox_inches="tight"
            )
            plt.close(fig)

    def get_solution_vector(self, pos, to_torch=True):
        """return a solution vector, from either side of the pareto extreme
        or the middle of the population
        for 2 objs:
            pos=0: f1 minimum (min Tsub)
            pos=1: compromise of the population
            pos=2: f2 minimum (min flux)
        for 3 objs:
            pos=0: f1 minimum (min Tsub)
            pos=1: f2 minimum (min zero flux)
            pos=2: f3 minimum (min max flux)
            pos=3: compromise of all
        """
        F = self.res.F
        if len(F[0]) == 2:
            if pos == 0:
                top_vector = self.res.X[np.argmin(F[:, 0])]
            elif pos == 1:
                top_vector = self.res.X[np.argmin(F[:, 1])]
            elif pos == 2:
                # top_vector = self.res.X[np.argmin(F.sum(axis=1))]
                top_vector = self.get_compromise_solution(path=ITER_PATH)

        elif len(F[0]) == 3:
            if pos == 0:
                top_vector = self.res.X[np.argmin(F[:, 0])]
            elif pos == 1:
                top_vector = self.res.X[np.argmin(F[:, 1])]
            elif pos == 2:
                top_vector = self.res.X[np.argmin(F[:, 2])]
            elif pos == 3:
                # top_vector = self.res.X[np.argmin(F.sum(axis=1))]
                top_vector = self.get_compromise_solution(path=ITER_PATH)

        if to_torch:
            return torch.tensor(top_vector).reshape(1, -1)
        else:
            return top_vector

    def get_hyperbolic_convergence(self, path=None):
        """docstring"""
        from pymoo.indicators.hv import Hypervolume

        approx_ideal = self.res.F.min(axis=0)
        approx_nadir = self.res.F.max(axis=0)
        metric = Hypervolume(
            ref_point=np.array([1.1] * len(self.res.F[0])),
            norm_ref_point=False,
            zero_to_one=True,
            ideal=approx_ideal,
            nadir=approx_nadir,
        )
        hv = [metric.do(_F) for _F in self.res.algorithm.callback.opt]
        n_evals = self.res.algorithm.callback.n_evals

        plt.figure(figsize=(7, 5))
        plt.plot(n_evals, hv, color="black", lw=0.7, label="Avg. CV of Pop")
        plt.scatter(n_evals, hv, facecolor="none", edgecolor="black", marker="p")
        # plt.title("Convergence")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Hypervolume")
        plt.savefig(path + "/hypervolume_convergence.png")
        plt.close()

    def get_compromise_solution(self, path=None):
        """Use achievement scalarization function, https://pymoo.org/mcdm/index.html
        input: pf, pareto front 1d array
        path: path to save the plot
        returns: solution vector and saves plot
        """
        from pymoo.decomposition.asf import ASF

        N_OBJ = self.res.F.shape[1]
        if N_OBJ == 3:
            weights = np.array([0.001, 0.001, 0.6])
        elif N_OBJ == 2:
            weights = np.array([0.0001, 0.5])
        decomp = ASF()
        self.decomp_idx = decomp(self.res.F, weights).argmin()
        print(
            f"Best regarding decomposition: Point {self.decomp_idx} - {self.res.F[self.decomp_idx]}"
        )
        print("Best with old method:", self.res.F[np.argmin(self.res.F.sum(axis=1))])

        return self.res.X[self.decomp_idx]

    def get_pareto_plot_with_cmaes(self, path, cmaes_obj):
        """
        Creates a Pareto plot (2D or 3D) and overlays the CMA-ES solution.
        """
        F = self.res.F.copy()

        if len(F[0]) == 2:
            plt.clf()
            fig = plt.figure(figsize=(10, 8))
            cmaes_obj_trans = cmaes_obj

            plt.scatter(
                F[:, 0],
                F[:, 1],
                s=30,
                facecolors="none",
                edgecolors="r",
                label="NSGA Solutions",
            )
            # Plot the CMA-ES point as a star marker:
            plt.scatter(
                cmaes_obj_trans[0],
                cmaes_obj_trans[1],
                s=100,
                facecolors="blue",
                edgecolors="black",
                marker="*",
                label="CMA-ES Solution",
            )
            plt.xlabel("Average temperature")
            plt.ylabel("Temperature std dev")
            plt.legend()
            save_path = os.path.join(path, "2D_pareto_front_with_cmaes.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved 2D Pareto plot with CMA-ES solution at {save_path}")


def main(config, ITER_PATH, pareto_dict=None):
    """main function"""

    # PRE-FENICS: get material grid for fenics plotting
    # ENVIRONMENT: use postprocess-env for this part (need pymoo + torch)

    PymooPost = PymooPostprocess(ITER_PATH)

    print("Saving pymoo post-processing...")
    PymooPost.get_hyperbolic_convergence(path=ITER_PATH)
    PymooPost.get_pareto_plot(
        path=ITER_PATH, plot_compromise=False, filter_outliers=False
    )


if __name__ == "__main__":
    ITER_PATH = "logs/_ONE_SOURCE_NSGA/test_nsga_10mpi_z16"
    config = os.path.join(ITER_PATH, "config.json")
    main(config, ITER_PATH)
