# image_processing.py
import numpy as np
# from mpi4py import MPI
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom, label, generate_binary_structure, binary_dilation, binary_erosion
# from skimage.morphology import binary_dilation, binary_erosion, remove_small_objects


def remove_small_specs(grid, min_size=3000):
    """
    Remove small specks in the material distribution.

    Parameters:
      grid (np.array): The input binary material distribution grid.
      min_size (int): Minimum size (in pixels) a connected component must have.

    Returns:
      np.array: The cleaned grid with small specks removed.
    """
    # Label all connected components (using 8-connectivity)
    structure = generate_binary_structure(2, 2)
    labeled, ncomponents = label(grid == 1, structure=structure)
    # Compute the size of each component
    component_sizes = np.bincount(labeled.ravel())
    # Identify components smaller than min_size
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled]
    # Remove small components (set them to background = 0)
    grid[too_small_mask] = 0
    return grid


def enforce_volume_fraction(img, vf, max_iter=50, atol=5e-2):
    """
    Adjust a binary image iteratively so that its volume fraction (proportion of 1's)
    approaches the desired value.

    Parameters:
      img (np.array): Input binary image (values 0 or 1).
      vf (float): Desired volume fraction (between 0 and 1).
      max_iter (int): Maximum number of iterations.
      atol (float): Absolute tolerance for convergence.

    Returns:
      np.array: Adjusted binary image with small specks removed.
    """
    # Ensure binary image is boolean for morphology operations.
    adjusted_img = img.copy().astype(np.bool_)
    current_vf = np.sum(adjusted_img) / adjusted_img.size
    niter = 0
    while (not np.isclose(current_vf, vf, atol=atol)) and (niter < max_iter):
        if current_vf < vf:
            # Increase volume fraction: dilate the image
            adjusted_img = binary_dilation(adjusted_img)
        elif current_vf > vf:
            # Decrease volume fraction: erode the image
            adjusted_img = binary_erosion(adjusted_img)
        current_vf = np.sum(adjusted_img) / adjusted_img.size
        niter += 1

    # Remove small specks after adjustment.
    # adjusted_img = remove_small_specs(adjusted_img.astype(np.uint8), min_size=1000)
    # Return as float32 array (with values 0 or 1)
    return adjusted_img.astype(np.float32)


def img_list_to_gamma_expression(img_list, config):
    """
    Build a function gamma_expression(x_input) -> array
    that, for each point x_input in the domain, returns
    0 or 1 depending on whether it lies in a region
    determined by 'img_list' or the top-extrusion mask.

    This handles both non-symmetric & symmetric geometries,
    with multiple source positions.
    """

    # ------------------------------------------------------------
    # 1) Define domain bounding box
    #    Note: If config.symmetry is True, we assume L_X is
    #    already half of the full domain width. The rest of
    #    the logic (0..L_X) stays the same.
    # ------------------------------------------------------------
    x_min = 0
    x_max = config.L_X
    y_min = 0
    y_max = config.L_Y + config.SOURCE_HEIGHT

    # Range in X and Y
    x_range = x_max - x_min
    y_range = y_max - y_min

    # For the images, we shift them upward by some offset:
    y_min_im = y_min + 1.5 * config.SOURCE_HEIGHT  # To make image higher

    # ------------------------------------------------------------
    # 2) Handle the source positions in absolute coordinates
    # ------------------------------------------------------------
    num_sources = len(config.source_positions)
    # Convert fractional positions [0..1] => [0..L_X]
    # If symmetry is enabled, we assume the sources are
    # already in the left half of the domain. We need to double
    # the source positions to cover the full domain.
    if config.symmetry:
        source_positions = np.array(config.source_positions) * config.L_X * 2
    else:
        source_positions = np.array(config.source_positions) * config.L_X

    # ------------------------------------------------------------
    # 3) Define how wide each image “strip” is
    #    (Your original approach: x_range / num_sources)
    # ------------------------------------------------------------
    image_mesh_width = x_range / num_sources

    # ------------------------------------------------------------
    # 4) Build an array of x-ranges for each image in img_list
    # ------------------------------------------------------------
    image_x_ranges = []
    for x_pos in source_positions:
        x_min_image = x_pos - image_mesh_width / 2
        x_max_image = x_pos + image_mesh_width / 2

        # Ensure x_min_image and x_max_image are within domain bounds
        x_min_image = max(x_min_image, x_min)
        x_max_image = min(x_max_image, x_max)

        image_x_ranges.append([x_min_image, x_max_image])

    # ------------------------------------------------------------
    # 5) Define the “bottom” of the top extrusion region
    #    This is for masking purposes.
    # ------------------------------------------------------------
    y_min_extrusion = config.L_Y - config.SOURCE_HEIGHT

    # ------------------------------------------------------------
    # 6) Construct the actual callback function for gamma
    # ------------------------------------------------------------
    def gamma_expression(x_input):
        """
        x_input.shape -> (gdim, N)
        Feturn an array of shape (N,) with 0 or 1.
        """
        # x_input is of shape (gdim, N)
        x_coords = x_input[0, :]
        y_coords = x_input[1, :]

        # Initialize gamma_values with zeros
        gamma_values = np.zeros_like(x_coords)

        # -------------------------------
        # A) “Paint” the images
        # -------------------------------
        for img, (x_min_image, x_max_image) in zip(img_list, image_x_ranges):
            # img is a 2D numpy array of shape (height, width) with 0/1
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

        # -------------------------------
        # B) Optional: mask extrusions
        # -------------------------------
        if config.mask_extrusion:
            # We do multiple extrusions for multiple source_positions
            # whether or not we are in symmetry mode.
            #
            # If symmetry = True, we typically have L_X “halved,” but
            # the logic is the same, as we still place extrusions
            # at each x_pos ± source_width/2 (with clamp).
            for x_pos in source_positions:
                # define a slightly bigger region than the width for safety
                half_w = 1.1 * config.SOURCE_WIDTH / 2.0
                x_min_ex = x_pos - half_w
                x_max_ex = x_pos + half_w

                # clamp to [0, L_X]
                if x_min_ex < x_min:
                    x_min_ex = x_min
                if x_max_ex > x_max:
                    x_max_ex = x_max

                # all points above y_min_extrusion, within x_min_ex..x_max_ex
                in_extrusion_x = np.logical_and(x_coords >= x_min_ex, x_coords <= x_max_ex)
                in_extrusion_y = y_coords > y_min_extrusion
                in_extrusion = np.logical_and(in_extrusion_x, in_extrusion_y)

                # set gamma=1 in that top region
                gamma_values[in_extrusion] = 1.0

        return gamma_values

    return gamma_expression


def generate_images(config, latent_vectors, model):
    # Generate image from latent vector
    img_list = []
    for z in latent_vectors:
        if config.blank:
            img = np.zeros((128, 128))
        else:
            # Ensure z is reshaped correctly if needed
            img = z_to_img(z.reshape(1, -1), model, config.vol_fraction)
        img = remove_small_specs(img, min_size=3000)
        img = enforce_volume_fraction(img, config.vol_fraction)
        img_list.append(img)

    # Apply symmetry to each image if enabled
    if config.symmetry:
        # only keep the left half of the image if source_position is 0.5
        for i, img in enumerate(img_list):
            if config.source_positions[i] == 0.5:
                img_list[i] = img[:, : img.shape[1] // 2]

    return img_list


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

