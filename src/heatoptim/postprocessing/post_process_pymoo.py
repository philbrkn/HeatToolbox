import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import json
# import sys
from mpi4py import MPI
import dolfinx.io

# for plotting fields:
from heatoptim.utilities.vae_module import load_vae_model
from heatoptim.utilities.image_processing import generate_images
from heatoptim.postprocessing.post_processing_gke import PostProcessingGKE
from heatoptim.solvers.mesh_generator import MeshGenerator
from heatoptim.config.sim_config import SimulationConfig
from heatoptim.solvers.solver_gke_module import GKESolver


class PymooPostprocess:
    """Post processing for pymoo plots
    e.g: pareto front, hypervolume convergence, running metric
    """
    def __init__(self, iter_path):
        self.iter_path = iter_path
        with open(iter_path + "/NSGA_Result.pk1", "rb") as f:
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
        plot_sols=False,
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

        plt.clf()
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(
            F[:, 0],
            F[:, 1],
            s=30,
            facecolors="none",
            edgecolors="black",
            label="Pareto Solutions",
        )
        plt.xlabel("Average temperature")
        plt.ylabel("Temperature std dev")
        # plt.title("Pareto Front")

        if plot_sols:
            plt.scatter(
                min(F[:, 0]),
                F[np.argmin(F[:, 0]), 1],
                s=150,
                facecolors="blue",
                edgecolors="blue",
                marker="*",
                label="Min avg. temp. solution",
            )
            plt.scatter(
                F[np.argmin(F[:, 1]), 0],
                min(F[:, 1]),
                s=150,
                facecolors="red",
                edgecolors="red",
                marker="*",
                label="Min temp. std. solution",
            )
            _, comp_pt = self.get_compromise_solution()
            # print(comp_pt)
            plt.scatter(
                comp_pt[0],
                comp_pt[1],
                s=150,
                facecolors="green",
                edgecolors="green",
                marker="*",
                label="Compromise solution",
            )
            plt.legend()
            plt.savefig(self.iter_path + "/2D_pareto_front_sols.png")
        else:
            plt.savefig(self.iter_path + "/2D_pareto_front.png")
        plt.close(fig)

    def get_solution_vector(self, pos):
        """return a solution vector, from either side of the pareto extreme
        or the middle of the population
        for 2 objs:
            pos=0: f1 minimum (min Tsub)
            pos=1: f2 minimum (min flux)
            pos=2: compromise of the population
        """
        F = self.res.F
        if pos == 0:
            top_vector = self.res.X[np.argmin(F[:, 0])]
        elif pos == 1:
            top_vector = self.res.X[np.argmin(F[:, 1])]
        elif pos == 2:
            top_vector, _ = self.get_compromise_solution()

        return top_vector

    def get_hyperbolic_convergence(self):
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
        plt.savefig(self.iter_path + "/hypervolume_convergence.png")
        plt.close()

    def get_compromise_solution(self):
        """Use achievement scalarization function, https://pymoo.org/mcdm/index.html
        input: pf, pareto front 1d array
        path: path to save the plot
        returns: solution vector and point
        """
        from pymoo.decomposition.asf import ASF

        N_OBJ = self.res.F.shape[1]
        if N_OBJ == 3:
            weights = np.array([0.001, 0.001, 0.6])
        elif N_OBJ == 2:
            weights = np.array([0.05, 0.5])
        decomp = ASF()
        self.decomp_idx = decomp(self.res.F, weights).argmin()
        print(
            f"Best decomposition: Point {self.decomp_idx} - {self.res.F[self.decomp_idx]}"
        )
        # print("Best with old method:", self.res.F[np.argmin(self.res.F.sum(axis=1))])

        return self.res.X[self.decomp_idx], self.res.F[self.decomp_idx]

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

    def visualize_fields(self, positions=[0, 1, 2], config_path=None):
        """Visualize gamma fields for selected positions on Pareto front."""
        with open(config_path, "r") as f:
            self.config_dict = json.load(f)
        self.config_dict["visualize"]["gamma"] = True
        self.config = SimulationConfig(self.config_dict)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.model = load_vae_model(self.rank, self.config.latent_size)

        for pos in positions:
            latent_vector = self.get_solution_vector(pos)

            # Generate image
            img_list = generate_images(self.config, [latent_vector], self.model)

            # Solve the heat problem with Fenicsx
            solver = self.run_fenics_solver(img_list=img_list)

            # Post-process results
            post_processor = PostProcessingGKE(self.rank, self.config)
            q, T = solver.U.sub(0).collapse(), solver.U.sub(1).collapse()
            V1, _ = solver.U.function_space.sub(1).collapse()

            # Plot gamma field
            # join with self.iter_oath
            if pos == 0:
                name_prefix = "minTAvg"
            elif pos == 1:
                name_prefix = "minTStd"
            else:
                name_prefix = "comp"
            gamma_field_filename = os.path.join(self.iter_path, f"{name_prefix}_")
            post_processor.postprocess_results(q, T, V=solver.W, msh=solver.msh, gamma=solver.gamma, fileadd=gamma_field_filename)

    def run_fenics_solver(self, img_list):
        mesh_gen = MeshGenerator(self.config)
        if self.config.symmetry:
            mesh_gen.sym_create_mesh()
        else:
            mesh_gen.create_mesh()
        # The mesh file must be available at this path
        msh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(
            "domain_with_extrusions.msh", MPI.COMM_SELF, gdim=2
        )
        solver = GKESolver(msh, facet_markers, self.config)
        solver.solve_image(img_list)
        return solver


def main(config_path, ITER_PATH, viz_sols=False):
    """main function"""

    PymooPost = PymooPostprocess(ITER_PATH)

    print("Saving pymoo post-processing...")
    PymooPost.get_hyperbolic_convergence()
    PymooPost.get_pareto_plot()
    PymooPost.get_pareto_plot(plot_sols=True)
    if viz_sols:
        PymooPost.visualize_fields(config_path=config_path)


if __name__ == "__main__":
    ITER_PATH = "logs/_ONE_SOURCE_NSGA2/test_nsga_10mpi_z16"
    viz_sols = True
    config_path = os.path.join(ITER_PATH, "config.json")
    main(config_path, ITER_PATH, viz_sols)
