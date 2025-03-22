# mesh_refinement_study.py

import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os


def main():
    # Define the length scale from your config or set it directly
    MEAN_FREE_PATH = 0.439e-6  # meters
    KNUDSEN = 1
    LENGTH = MEAN_FREE_PATH / KNUDSEN  # meters

    # List of mesh resolutions to test (from coarser to finer)
    resolutions = [
        LENGTH / 0.1,    # Coarser mesh
        LENGTH / 0.5,    # Coarser mesh
        LENGTH / 1,    # Coarser mesh
        LENGTH / 2,
        LENGTH / 3,
        LENGTH / 5,    # Default mesh
        LENGTH / 8,
        LENGTH / 10,
        LENGTH / 12,
        LENGTH / 15,
        LENGTH / 20,
        LENGTH / 30,    # Finer mesh
        # LENGTH / 40,    # Finer mesh
    ]

    # Number of MPI processes per simulation
    mpi_processes = 1  # Adjust based on your system

    # Initialize a list to store average temperatures
    avg_temps = []

    for res in resolutions:
        # Construct the command with mpirun
        command = [
            "python3",
            "heatoptim.main",
            "--res",
            str(res),
            "--visualize",
            "none",
            "--sym",
            # "--blank",  # Uncomment if needed
        ]

        print(f"\nRunning simulation with resolution: {res:.2e} m")
        
        # Define log file name
        log_file = f"simulation_res_{res:.2e}.log"
        
        try:
            with open(log_file, "w") as f:
                result = subprocess.run(
                    command,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True  # Raise an error if the simulation fails
                )
        except subprocess.CalledProcessError as e:
            print(f"Simulation failed at resolution {res:.2e} m.")
            print(f"Check log file {log_file} for details.")
            avg_temps.append((res, None))
            continue

        # Read the log file to extract average temperature
        avg_temp = None
        with open(log_file, "r") as f:
            for line in f:
                if "Average temperature" in line:
                    try:
                        temp_str = line.split(":")[1].strip().split()[0]
                        avg_temp = float(temp_str)
                        avg_temps.append((res, avg_temp))
                    except (IndexError, ValueError):
                        print(f"Error parsing temperature from line: {line}")
                        avg_temps.append((res, None))
                    break
            else:
                print(f"Average temperature not found in log file for resolution {res:.2e} m.")
                avg_temps.append((res, None))

        print(f"Average Temperature for resolution {res:.2e} m: {avg_temp} K")

        # Optionally, delete the log file after parsing to save space
        # os.remove(log_file)

    # After running all simulations, print and save the results
    print("\nMesh Refinement Study Results:")
    for res, temp in avg_temps:
        if temp is not None:
            print(f"Resolution: {res:.2e} m, Average Temperature: {temp:.6f} K")
        else:
            print(f"Resolution: {res:.2e} m, Average Temperature: N/A")

    # Convert resolutions and temperatures to numpy arrays for plotting
    res_array = np.array([res for res, temp in avg_temps if temp is not None])
    temp_array = np.array([temp for res, temp in avg_temps if temp is not None])

    # Plot average temperature vs. mesh resolution
    plt.figure(figsize=(8, 6))
    plt.loglog(res_array, temp_array, marker='o', linestyle='-')
    plt.xlabel('Mesh Resolution (m)')
    plt.ylabel('Average Temperature (K)')
    plt.title('Mesh Refinement Study: Average Temperature vs. Mesh Resolution')
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis()  # Finer meshes (smaller resolutions) on the right
    plt.tight_layout()
    plt.savefig("mesh_refinement_study_results.png")
    plt.show()


if __name__ == "__main__":
    main()
