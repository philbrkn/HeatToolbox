import os
import subprocess
import tkinter as tk
from tkinter import messagebox


def submit_job(hpc_user, hpc_host, hpc_path, log_dir, password):
    local_base_path = "."
    # Transfer the log directory to the HPC
    transfer_files_to_hpc(
        local_base_path, log_dir, hpc_user, hpc_host, hpc_path, password
    )

    try:
        # SSH command with environment setup for qsub
        ssh_command = [
            "sshpass",
            "-p",
            password,
            "ssh",
            f"{hpc_user}@{hpc_host}",
            f"bash -l -c 'cd {hpc_path} && qsub {log_dir}/hpc_run.sh'",
        ]

        # Run the command
        result = subprocess.run(ssh_command, check=True, text=True, capture_output=True)

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

    # Use subprocess with `sshpass` for password-based file transfer
    rsync_command = [
        "sshpass",
        "-p",
        password,
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
        subprocess.run(rsync_command, check=True)
        print(
            f"Successfully transferred to {hpc_user}@{hpc_host}:{hpc_remote_base_path}"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"File transfer failed: {e}")


def prompt_password(prompt="Enter HPC Password"):
    """Prompt the user to enter their HPC password."""
    password_window = tk.Toplevel()
    password_window.title(prompt)

    tk.Label(password_window, text=prompt).pack(pady=5)

    password_var = tk.StringVar()
    password_entry = tk.Entry(password_window, show="*", textvariable=password_var)
    password_entry.pack(pady=5)
    password_entry.focus_set()

    def on_submit():
        password_window.destroy()

    tk.Button(password_window, text="Submit", command=on_submit).pack(pady=5)

    # Wait for the user to enter the password
    password_window.wait_window()

    return password_var.get()


def save_hpc_script(script_content, log_dir):
    """Save the HPC script to the specified log directory."""
    hpc_path = os.path.join(log_dir, "hpc_run.sh")
    with open(hpc_path, "w") as f:
        f.write(script_content)
    print(f"HPC script saved to {hpc_path}")
    return hpc_path
