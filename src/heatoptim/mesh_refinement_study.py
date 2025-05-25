# mesh_refinement_study.py

import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os


def main():
    # folder name:
    folder_name = "mesh_res_study_logs_blank"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # List of mesh resolutions to test (from coarser to finer)
    resolutions = [
        0.1,    # Coarser mesh
        0.5,    # Coarser mesh
        1,    # Coarser mesh
        2,
        3,
        5,    # Default mesh
        8,
        10,
        12,
        15,
        20,
        30,    # Finer mesh
    ]

    # Initialize a list to store average temperatures
    avg_temps = []

    for res in resolutions:
        # need to create a config file and change the resolution for each and pass it in as arg
        # Create a config dictionary for the current resolution
        # Construct the command with mpirun
        command = [
            "python3",
            "-m",
            "heatoptim.main",
            "--config",
            f"{folder_name}/config.json",  # Path to config file
            "--res",
            str(res),
        ]

        # Define log file name
        log_file = f"{folder_name}/simulation_res_{res:.2e}.log"

        print(f"\nRunning simulation with resolution: {res:.2e} m")
        # try:
        #     with open(log_file, "w") as f:
        #         result = subprocess.run(
        #             command,
        #             stdout=f,
        #             stderr=subprocess.STDOUT,
        #             text=True,
        #             check=True  # Raise an error if the simulation fails
        #         )
        # except subprocess.CalledProcessError as e:
        #     print(f"Simulation failed at resolution {res:.2e} m.")
        #     print(f"Check log file {log_file} for details.")
        #     avg_temps.append((res, None))
        #     continue

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
    LENGTH = 5e-07
    for res, temp in avg_temps:
        if temp is not None:
            print(f"Resolution: {res:.2e} m, Average Temperature: {temp:.6f} K")
        else:
            print(f"Resolution: {res:.2e} m, Average Temperature: N/A")

    # Convert resolutions and temperatures to numpy arrays for plotting
    # res_array = np.array([LENGTH/res for res, temp in avg_temps if temp is not None])
    res_array = np.array([LENGTH/res for res, temp in avg_temps if temp is not None])
    temp_array = np.array([temp for res, temp in avg_temps if temp is not None])

    # Plot average temperature vs. mesh resolution
    plt.figure(figsize=(8, 6))
    plt.loglog(res_array, temp_array, marker='o', linestyle='-', color='blue')
    # plt.plot(res_array, temp_array, marker='o', linestyle='-')
    plt.xlabel('Mesh Resolution (m)')
    plt.ylabel('Average Temperature (K)')
    plt.title('Mesh Refinement Study: Average Temperature vs. Mesh Resolution')
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis()  # Finer meshes (smaller resolutions) on the right
    plt.tight_layout()
    plt.savefig(f"{folder_name}/mesh_refinement_study_results.png")
    plt.show()


if __name__ == "__main__":
    main()
