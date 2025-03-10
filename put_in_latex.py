import os
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import shutil


# Assuming your cropping function is like this
def crop_image(image_path, output_path, crop_rectangle):
    """
    Crops an image and displays it using matplotlib.
    
    :param image_path: The path to the image to be cropped.
    :param crop_rectangle: A tuple of (left, upper, right, lower) pixel coordinates.
    """

    # Open the image file
    with Image.open(image_path) as img:
        if crop_rectangle[2] == -1:
            crop_rectangle[2] = img.size[0]
        if crop_rectangle[3] == -1:
            crop_rectangle[3] = img.size[1]

        # Crop the image
        cropped_img = img.crop(crop_rectangle)

        # Display the cropped image using matplotlib
        fig, ax = plt.subplots()
        ax.imshow(cropped_img)
        ax.axis('off')  # Hide axes

        # Remove all the axes including the figure border and padding, and save the image
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free up memory


def add_legend_image(image_path, legend_path, output_path, crop_rectangle=(50, 100, 460, 645)):
    '''add legend image to bottom of material image'''
    # Open the image
    with Image.open(image_path) as img:
        # concatenate the legend image to the bottom of the material image
        with Image.open(legend_path) as legend:
            original_width, original_height = img.size
            crop_width, crop_height = crop_rectangle[2] - crop_rectangle[0], crop_rectangle[3] - crop_rectangle[1]

            # Calculate the new height to maintain the aspect ratio
            new_height = int(original_width * (crop_height / crop_width))
            padding = new_height - original_height
            # resize legend to match padding
            legend = legend.resize((original_width, padding))

            # Create a new image with the same width and the new height
            new_img = Image.new("RGB", (int(1.1*original_width), new_height), color=(255, 255, 255))

            new_img.paste(img, (0, 0))
            new_img.paste(legend, (0, original_height))
            new_img.save(output_path)


def add_padding(image_path, output_path, crop_rectangle):
    # Open the image
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        crop_width, crop_height = crop_rectangle[2] - crop_rectangle[0], crop_rectangle[3] - crop_rectangle[1]

        # Calculate the new height to maintain the aspect ratio
        new_height = int(original_width * (crop_height / crop_width))
        padding = new_height - original_height

        # Create a new image with the same width and the new height
        padded_img = Image.new("RGB", (original_width, new_height), color=(255, 255, 255))

        # Paste the original image onto the padded image
        padded_img.paste(img, (0, 0))

        # Save the padded image
        padded_img.save(output_path)


# Function to generate LaTeX include commands
def generate_latex_include_commands(subfigures_dict, image_folder, caption):
    latex_commands = []
    folder = os.path.basename(image_folder)
    for subfigure_group, file_list in subfigures_dict.items():
        subfigures = []
        for image_file in file_list:
            path_parts = image_file.split(os.sep)[-3:]
            relative_path = os.path.join(*path_parts)
            subfigure_label = os.path.splitext(os.path.basename(image_file))[0].replace('_', '')

            # how much space the image takes:
            pct_space = round(1/len(file_list), 2)
            mat_space = round(0.924 * pct_space, 3)
            tmp_space = round(0.84 * pct_space, 3)
            name = os.path.splitext(os.path.basename(image_file))[0]
            # Creating the subfigure
            if name == "material_image_compromise" or name == "3D_pareto_front":  # for the material image
                subfigure = f"\\begin{{subfigure}}{{{mat_space}\\textwidth}}\n"
            elif name == "TEMP_DIST":
                subfigure = f"\\begin{{subfigure}}{{{tmp_space}\\textwidth}}\n"
            else:
                subfigure = f"\\begin{{subfigure}}{{{pct_space}\\textwidth}}\n"
            subfigure += (f"\\centering\n"
                        f"\\includegraphics[width=\\linewidth]{{{relative_path}}}\n"
                        "\\caption{caption}\n"  # add caption here
                        f"\\label{{fig:{subfigure_label}_{folder}}}\n"
                        "\\end{subfigure}%\n")
            subfigures.append(subfigure)
        # Join the subfigures, add the main figure environment, and remove the % from the last subfigure
        subfigures[-1] = subfigures[-1].rstrip('%\n') + "\n"  # Remove the '%' from the last subfigure
        figure_env = (
            "\\begin{figure*}[tp]\n"
            "    \\centering\n"
            f"    {''.join(subfigures)}"
            "    \\caption{Your caption here}\n"
            f"    \\label{{fig:{subfigure_group}}}\n"
            "\\end{figure*}\n"
        )
        latex_commands.append(figure_env)

    return latex_commands


# Function to process images and generate a LaTeX file
def process_images(log_directory, git_repo_path, fig_folder_name, subfigures_mapping, caption="Your caption here"):
    figures_dir = os.path.join(git_repo_path, "figs", fig_folder_name)
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    image_files_to_add = {}

    for group_name, file_patterns in subfigures_mapping.items():
        image_files_to_add[group_name] = []
        for file_pattern in file_patterns:
            for file_name in os.listdir(log_directory):
                if file_name.startswith(file_pattern) and file_name.endswith((".png", ".jpg", ".jpeg")):
                    if file_name.startswith("plot"):
                        # Crop images starting with "plot"
                        input_path = os.path.join(log_directory, file_name)
                        output_path = os.path.join(figures_dir, file_name)
                        crop_image(input_path, output_path, crop_rectangle=(50, 100, 460, 580))
                        image_files_to_add[group_name].append(output_path)
                    elif file_name == "material_image_compromise.jpg":
                        # Copy the compromise image and add padding
                        source_path = os.path.join(log_directory, file_name)
                        dest_path = os.path.join(figures_dir, file_name)
                        # add_padding(source_path, dest_path, crop_rectangle=(50, 100, 460, 645))
                        legend_path = "post_process/material_legend.png"
                        add_legend_image(source_path, legend_path, dest_path)
                        # shutil.copy2(source_path, dest_path)
                        image_files_to_add[group_name].append(dest_path)
                    elif file_name == "3D_pareto_front.png":
                        # Crop images starting with "plot"
                        input_path = os.path.join(log_directory, file_name)
                        output_path = os.path.join(figures_dir, file_name)
                        crop_image(input_path, output_path, crop_rectangle=[0, 50, -1, -1])
                        image_files_to_add[group_name].append(output_path)
                    else:
                        # Copy the compromise image without cropping
                        source_path = os.path.join(log_directory, file_name)
                        dest_path = os.path.join(figures_dir, file_name)
                        shutil.copy2(source_path, dest_path)
                        image_files_to_add[group_name].append(dest_path)

    # Generate LaTeX file content
    latex_commands = generate_latex_include_commands(image_files_to_add, figures_dir, caption)
    latex_content = "\n".join(latex_commands)

    # Write LaTeX commands to a file
    latex_file_path = os.path.join(figures_dir, "figures.tex")
    with open(latex_file_path, 'w') as latex_file:
        latex_file.write(latex_content)

    return latex_file_path, image_files_to_add


def reset_conflicted_to_remote(git_repo_path):
    os.chdir(git_repo_path)
    subprocess.run(['git', 'fetch'])  # Ensure the local repo knows about the latest remote state

    # Detect any unmerged files
    result = subprocess.run(['git', 'ls-files', '-u'], stdout=subprocess.PIPE)
    conflicts = result.stdout.decode().strip()
    if conflicts:
        print("Detected unmerged files. Resolving by favoring the remote versions.")
        conflicted_files = set()
        for line in conflicts.split('\n'):
            parts = line.split('\t')
            if len(parts) > 2:
                conflicted_files.add(parts[3])  # File path is the fourth element (index 3)

        if conflicted_files:
            print("Resolving conflicts for:", conflicted_files)
            for file in conflicted_files:
                # Force use of the remote version for each conflicted file
                subprocess.run(['git', 'checkout', '--theirs', file])
                subprocess.run(['git', 'add', file])

        # After handling all files, ensure the index is clean
        if subprocess.run(['git', 'commit', '-m', 'Resolved conflicts by favoring remote versions']).returncode != 0:
            print("Error: Committing resolved files failed. Please check the repository state manually.")

def push_git(git_repo_path, latex_file_path, image_files_to_add, IMAGES_ONLY=False):
    reset_conflicted_to_remote(git_repo_path)

    # Flatten the list of image files to add
    all_image_files_to_add = []
    for file_group in image_files_to_add.values():
        all_image_files_to_add.extend(file_group)

    # Adding files to staging area
    if IMAGES_ONLY:
        subprocess.run(['git', 'add'] + all_image_files_to_add)
    else:
        subprocess.run(['git', 'add', latex_file_path] + all_image_files_to_add)

    # Commit and push changes
    commit_result = subprocess.run(['git', 'commit', '-m', 'Update images and LaTeX file'])
    if commit_result.returncode != 0:
        print("Nothing to commit, working tree clean")
    else:
        push_result = subprocess.run(['git', 'push'])
        if push_result.returncode != 0:
            print("Git push failed.")
            exit(1)


# Run the process_images function and push to git
def main():
    IMAGES_ONLY = False
    git_repo_name = "66fdbdbbd69a7de183efaa4f"  # REVISED OVERLEAF
    # this git repo is in the current working directory
    git_repo_path = os.path.join(os.getcwd(), git_repo_name)

    # FOR RUN RESULTS #
    # log_directory = "logs/vf02/zm/20231107_1227_4cells_zm"
    # log_directory = "logs/vf02/mz/20231112_2254_4cells_mz"
    # log_directory = "logs/vf02/mz/20231113_0520_12cells_mz"
    # log_directory = "logs/vf02/zm/20231107_1301_6cells_zm"
    # fig_folder_name = "vf02_" + log_directory.split('_')[2] + '_' + log_directory.split('_')[3]
    # subfigures_mapping = {
    #     f"{fig_folder_name}_FENICS": ['material_image_compromise', 'TEMP_DIST',
    #                                   'plot_flux_vector_field', 'plot_tsub_distribution'],
    #     f"{fig_folder_name}_OPTIM": ['3D_pareto_front', 'hypervolume_convergence']
    # }

    # PDMS ONLY #
    # log_directory = "logs/PDMS_only"
    # fig_folder_name = "PDMS_only"
    log_directory = "logs/paper/Fe_only"
    fig_folder_name = "Fe_only"
    # log_directory = "logs/Cu_only"
    # fig_folder_name = "Cu_only"
    # log_directory = "logs/paper/6cellsC"
    # fig_folder_name = "6cellsC"
    subfigures_mapping = {
         f"{fig_folder_name}_FENICS": ['material_image_compromise', 'TEMP_DIST',
                                      'plot_flux_vector_field', 'plot_tsub_distribution']}
    # caption = "Selected 4-cells candidate representing trade-off multi-objective performance using ASF for the insulator-concentrator setup: a) material distribution of Fe, Cu, and PDMS. b) temperature distribution with isotherms over the multi-material domain c) heat flux vector field interpolated onto a coarser mesh for visibility purposes. d) perturbation in the solved temperature field against the field corresponding to a uniform Fe (background)."
    
    # FOR ASSEMBLY PICS #
    # log_directory = "images/paper_assembly"
    # fig_folder_name = "paper_assembly"
    # subfigures_mapping = {
    #     'first_set': ['paper_assembly_1.png', 'paper_assembly_2.png', 'paper_assembly_3.png',
    #                   'paper_assembly_4.png', 'paper_assembly_5.png']
    # }

    # FOR MIDLINE RESULTS #
    # log_directory = os.getcwd()
    # fig_folder_name = "vf02_mz"
    # subfigures_mapping = {
    #     fig_folder_name: ['vf02_mz_bar_plot.png', 'vf02_mz_temp_line.png']
    # }

    # log_directory = os.getcwd()
    # fig_folder_name = "vf02_zm"
    # subfigures_mapping = {
    #     fig_folder_name: ['vf02_zm_bar_plot.png', 'vf02_zm_temp_line.png']
    # }

    latex_file_path, image_files_to_add = process_images(log_directory, git_repo_path,
                                                         fig_folder_name, subfigures_mapping)

    push_git(git_repo_path, latex_file_path, image_files_to_add, IMAGES_ONLY)


if __name__ == "__main__":
    main()
