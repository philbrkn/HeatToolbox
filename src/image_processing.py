# image_processing.py
import numpy as np
# from mpi4py import MPI
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom


def img_list_to_gamma_expression(img_list, config):
    # Compute global min and max coordinates
    x_min = 0
    x_max = config.L_X
    y_min = 0
    y_max = config.L_Y + config.SOURCE_HEIGHT

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Adjust y_min for image mapping
    y_min_im = y_min + 1.5 * config.SOURCE_HEIGHT  # To make image higher

    num_sources = len(config.source_positions)
    source_positions = np.array(config.source_positions) * config.L_X

    # Define image mesh width
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

    # Define y-coordinate for the start of extrusion
    y_min_extrusion = config.L_Y - config.SOURCE_HEIGHT

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
            in_image = np.logical_and(
                x_coords >= x_min_image, x_coords <= x_max_image
            )

            # Map y as well, assuming the image spans from y_min to y_max
            in_image = np.logical_and(
                in_image,
                np.logical_and(y_coords >= y_min, y_coords <= y_max)
            )

            # Normalize x and y within the image's range
            x_norm = (x_coords[in_image] - x_min_image) / image_mesh_width
            y_norm = (y_coords[in_image] - y_min_im) / y_range

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

        # Apply mask extrusion
        if config.mask_extrusion:
            # Mask the top extrusion if requested
            if config.symmetry:
                x_min_extrusion = config.L_X - config.SOURCE_WIDTH
                x_max_extrusion = config.L_X

                in_extrusion = np.logical_and(
                    np.logical_and(
                        y_coords > y_min_extrusion, x_coords >= x_min_extrusion
                    ),
                    x_coords <= x_max_extrusion,
                )
                gamma_values[in_extrusion] = 1.0
            else:
                for x_pos in source_positions:
                    x_min_extrusion = x_pos - 1.1 * config.SOURCE_WIDTH / 2
                    x_max_extrusion = x_pos + 1.1 * config.SOURCE_WIDTH / 2

                    in_extrusion = np.logical_and(
                        np.logical_and(
                            y_coords > y_min_extrusion, x_coords >= x_min_extrusion
                        ),
                        x_coords <= x_max_extrusion,
                    )
                    gamma_values[in_extrusion] = 1.0

        return gamma_values

    return gamma_expression


def gaussian_blur(img, sigma=1):
    """Apply Gaussian blur using PyTorch."""
    # Create a 2D Gaussian kernel
    kernel_size = int(4 * sigma + 1)
    x = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    gauss_kernel = gauss[:, None] * gauss[None, :]
    gauss_kernel = gauss_kernel.expand(
        1, 1, *gauss_kernel.shape
    )  # Shape to match convolution input

    # Add batch and channel dimensions to the image for convolution
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    # Apply 2D convolution for Gaussian blur
    blurred_img = F.conv2d(img_tensor, gauss_kernel, padding=kernel_size // 2)

    # Remove extra dimensions
    return blurred_img.squeeze().numpy()


def apply_volume_fraction(img, vf):
    # Flatten the image to a 1D array
    img_flat = img.flatten()
    # Sort the pixels
    sorted_pixels = np.sort(img_flat)
    # Determine the threshold that will result in the desired volume fraction
    num_pixels = img_flat.size
    k = int((1 - vf) * num_pixels)
    threshold = (sorted_pixels[k] + sorted_pixels[k - 1]) / 2.0
    # Apply the threshold
    img_binary = (img >= threshold).astype(np.float32)
    return img_binary


def upsample_image(img, zoom_factor):
    return zoom(img, zoom_factor, order=3)  # Cubic interpolation


def z_to_img(z, model, vf, device=torch.device("cpu"), sigma=1.5, zoom_factor=3):
    z = torch.from_numpy(z).float().unsqueeze(0).to(device)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        sample = model.decode(z)
        # Remove batch and channel dimensions
        img = sample.squeeze().squeeze().cpu().numpy()
        # Flip the image
        img = img[::-1, :]
        # Take the left half of the image and resymmetrize
        img = img[:, : img.shape[1] // 2]
        img = np.concatenate((img, img[:, ::-1]), axis=1)

    # Apply Gaussian blur to smoothen the image
    img_smoothed = gaussian_blur(img, sigma=sigma)
    # Upsample the image before thresholding
    img_upsampled = upsample_image(img_smoothed, zoom_factor=zoom_factor)

    # Apply volume fraction control or default thresholding
    if vf is None:
        # Default binary thresholding
        img_binary = (img_upsampled >= 0.5).astype(np.float32)
    else:
        img_binary = apply_volume_fraction(img_upsampled, vf)

    return img_binary


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
