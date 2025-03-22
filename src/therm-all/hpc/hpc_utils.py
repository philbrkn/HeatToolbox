import os
import subprocess
import tkinter as tk
from tkinter import messagebox
import time


def submit_job(hpc_user, hpc_host, hpc_path, log_dir, password):

    local_base_path = "."
    # Transfer the log directory to the HPC
    transfer_files_to_hpc(
        local_base_path, log_dir, hpc_user, hpc_host, hpc_path, password
    )
    wait_for_files(
        hpc_user,
        hpc_host,
        os.path.join(hpc_path, log_dir),
        ["hpc_run.sh", "config.json"],
        password
    )
    try:
        # Set the SSHPASS environment variable
        env = os.environ.copy()
        env['SSHPASS'] = password

        # SSH command with environment setup for qsub
        ssh_command = [
            "sshpass",
            "-e",  # Use the password from the environment variable
            "ssh",
            f"{hpc_user}@{hpc_host}",
            f"bash -l -c 'cd {hpc_path} && qsub {log_dir}/hpc_run.sh'",
        ]

        # Run the command
        result = subprocess.run(ssh_command, check=True, text=True, capture_output=True, env=env)

        # Extract the job ID from the result
        job_id = result.stdout.strip()
        messagebox.showinfo("Success", f"Job submitted successfully!\nJob ID: {job_id}")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Job submission failed.\n{e.stderr}")


def transfer_files_to_hpc(
    local_base_path, log_dir, hpc_user, hpc_host, hpc_remote_base_path, password
):
    """
    Transfer files to HPC while maintaining the folder structure.

    Args:
        local_base_path (str): The local base path (e.g., "BTE-NO").
        log_dir (str): The log directory to transfer (relative to the local base path, e.g., "logs/logfolderiwant").
        hpc_user (str): HPC username.
        hpc_host (str): HPC hostname.
        hpc_remote_base_path (str): The remote base path on the HPC (e.g., "~/BTE-NO").
        password (str): Password for HPC access.
    """
    # Ensure local paths exist
    local_src_path = os.path.join(local_base_path, "src")
    local_log_path = os.path.join(local_base_path, log_dir)
    if not os.path.exists(local_src_path):
        raise FileNotFoundError(f"Source directory not found: {local_src_path}")
    if not os.path.exists(local_log_path):
        raise FileNotFoundError(f"Log directory not found: {local_log_path}")

    # Set the SSHPASS environment variable
    env = os.environ.copy()
    env['SSHPASS'] = password

    # Use subprocess with `sshpass` for password-based file transfer
    rsync_command = [
        "sshpass", "-e",
        "rsync",
        "-avz",
        "--progress",
        "--relative",
        "--exclude-from",
        os.path.join(local_base_path, "hpc_exclude.txt"),
        f"{local_src_path}/",  # Transfer the src directory
        f"{local_log_path}/",  # Transfer the specific log directory
        f"{hpc_user}@{hpc_host}:{hpc_remote_base_path}/",
    ]
    print(rsync_command)

    # Execute the command
    try:
        subprocess.run(rsync_command, check=True, env=env)
        print(
            f"Successfully transferred to {hpc_user}@{hpc_host}:{hpc_remote_base_path}"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"File transfer failed: {e}")


def wait_for_files(hpc_user, hpc_host, remote_path, files, password, timeout=30, interval=2):
    env = os.environ.copy()
    env['SSHPASS'] = password
    elapsed = 0
    while elapsed < timeout:
        try:
            # Check if all files exist on HPC
            check_command = [
                "sshpass", "-e",
                "ssh", f"{hpc_user}@{hpc_host}",
                f"ls {' '.join([os.path.join(remote_path, file) for file in files])}"
            ]
            subprocess.run(check_command, check=True, env=env)
            return  # Files are available
        except subprocess.CalledProcessError:
            elapsed += interval
            print(f"Waiting for files to sync... ({elapsed}/{timeout} seconds)")
            time.sleep(interval)
    raise RuntimeError("Files did not appear on HPC within the timeout period.")


def prompt_password(prompt="Enter HPC Password"):
    """Prompt the user to enter their HPC password securely."""
    import getpass
    password = getpass.getpass(prompt + ": ")
    return password


def save_hpc_script(script_content, log_dir):
    """Save the HPC script to the specified log directory."""
    hpc_path = os.path.join(log_dir, "hpc_run.sh")
    with open(hpc_path, "w") as f:
        f.write(script_content)
    print(f"HPC script saved to {hpc_path}")
    return hpc_path
