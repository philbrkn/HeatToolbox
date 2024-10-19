# image_processing.py
import numpy as np
from mpi4py import MPI


def img_list_to_gamma_expression(img_list, config):
    # Compute global min and max coordinates
    x_min = 0
    x_max = config.L_X
    y_min = 0
    y_max = config.L_Y + config.SOURCE_HEIGHT

    x_range = x_max - x_min
    y_range = y_max - y_min

    y_min_im += 1.5 * config.SOURCE_HEIGHT  # To make image higher

    num_sources = len(config.source_positions)
    source_positions = np.array(config.source_positions) * config.L_X

    # image mesh width
    image_mesh_width = x_range / num_sources

    # Calculate the physical x ranges of each image in the domain
    image_x_ranges = []
    for x_pos in source_positions:
        x_min_image = x_pos - image_mesh_width / 2
        x_max_image = x_pos + image_mesh_width / 2

        # Ensure x_min_image and x_max_image are within domain bounds
        x_min_image = max(x_min_image, x_min)
        x_max_image = min(x_max_image, x_max)

        image_x_ranges.append([x_min_image, x_max_image])

    def gamma_expression(x_input):
        # x_input is of shape (gdim, N)
        x_coords = x_input[0, :]
        y_coords = x_input[1, :]

        # Initialize gamma_values with zeros
        gamma_values = np.zeros_like(x_coords)

        # For each image, map it onto the domain
        for img, (x_min_image, x_max_image) in zip(img_list, image_x_ranges):
            img_height, img_width = img.shape

            # Determine which points are within the image's x-range
            in_image = np.logical_and(x_coords >= x_min_image, x_coords <= x_max_image)

            # Map y as well, assuming the image spans from y_min to y_max
            in_image = np.logical_and(
                in_image, np.logical_and(y_coords >= y_min, y_coords <= y_max)
            )

            # Normalize x and y within the image's range
            x_norm = (x_coords[in_image] - x_min_image) / (x_max_image - x_min_image)
            y_norm = (y_coords[in_image] - y_min_im) / (y_max - y_min)
            # check number of these coords:
            print("x_norm", len(x_norm))
            print("y_norm", len(y_norm))
            x_indices = np.clip(
                (x_norm * (img_width - 1)).astype(int), 0, img_width - 1
            )
            y_indices = np.clip(
                ((1 - y_norm) * (img_height - 1)).astype(int), 0, img_height - 1
            )

            # Get gamma values from the image
            gamma_values_in_image = img[y_indices, x_indices]

            # Update gamma_values array:
            # if gamma_values_in_image == 1, set gamma_values to 1
            gamma_values_current = gamma_values[in_image]
            gamma_values_new = np.where(
                gamma_values_in_image == 1, 1, gamma_values_current
            )
            gamma_values[in_image] = gamma_values_new

        return gamma_values

    return gamma_expression


def img_to_gamma_expression(img, config):
    # Compute global min and max coordinates
    x_min = 0
    x_max = config.L_X
    y_min = 0
    y_max = config.L_Y + config.SOURCE_HEIGHT

    img_height, img_width = img.shape

    x_range = x_max - x_min
    y_range = y_max - y_min

    y_min += 1.5 * config.SOURCE_HEIGHT  # To make image higher

    # Avoid division by zero in case the mesh or image has no range
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0

    y_min_extrusion = config.L_Y - config.SOURCE_HEIGHT
    # Define the extrusion region
    if config.symmetry:
        x_min_extrusion = config.L_X - config.SOURCE_WIDTH
        x_max_extrusion = config.L_X
    elif len(config.source_positions) > 1:
        x_min_extrusion_l = config.L_X * 0.25 - config.SOURCE_WIDTH / 2
        x_max_extrusion_l = config.L_X * 0.25 + config.SOURCE_WIDTH / 2
        x_min_extrusion_r = config.L_X * 0.75 - config.SOURCE_WIDTH / 2
        x_max_extrusion_r = config.L_X * 0.75 + config.SOURCE_WIDTH / 2
    else:
        x_min_extrusion = config.L_X / 2 - config.SOURCE_WIDTH / 2
        x_max_extrusion = config.L_X / 2 + config.SOURCE_WIDTH / 2

    def gamma_expression(x_input):
        # x_input is of shape (gdim, N)
        x_coords = x_input[0, :]
        y_coords = x_input[1, :]

        # Initialize gamma_values with zeros
        gamma_values = np.zeros_like(x_coords)

        # For all points in the mesh, scale the image to the full size of the mesh
        x_norm = (x_coords - x_min) / x_range  # Normalize x coordinates
        y_norm = (y_coords - y_min) / y_range  # Normalize y coordinates
        # check length of these coords:
        print("x_norm", len(x_norm))
        print("y_norm", len(y_norm))
        x_indices = np.clip((x_norm * (img_width - 1)).astype(int), 0, img_width - 1)
        y_indices = np.clip(
            ((1 - y_norm) * (img_height - 1)).astype(int), 0, img_height - 1
        )

        gamma_values = img[y_indices, x_indices]  # Map image values to the mesh

        if config.mask_extrusion:
            # Mask the top extrusion if requested
            if len(config.source_positions) > 1:
                # use the left and the right
                in_extrusion_l = np.logical_and(
                    np.logical_and(
                        y_coords > y_min_extrusion, x_coords >= x_min_extrusion_l
                    ),
                    x_coords <= x_max_extrusion_l,
                )
                in_extrusion_r = np.logical_and(
                    np.logical_and(
                        y_coords > y_min_extrusion, x_coords >= x_min_extrusion_r
                    ),
                    x_coords <= x_max_extrusion_r,
                )
                gamma_values[in_extrusion_l] = 1.0
                gamma_values[in_extrusion_r] = 1.0
            else:
                in_extrusion = np.logical_and(
                    np.logical_and(
                        y_coords > y_min_extrusion, x_coords >= x_min_extrusion
                    ),
                    x_coords <= x_max_extrusion,
                )
                gamma_values[in_extrusion] = 1.0

        return gamma_values

    return gamma_expression
