import os
import subprocess
import tkinter as tk
from tkinter import messagebox


def submit_job(hpc_user, hpc_host, log_dir, password):
    hpc_remote_path = "~/BTE-NO"

    # Transfer the log directory to the HPC
    transfer_files_to_hpc(log_dir, remote_path, hpc_user, hpc_host, password)

    try:
        # SSH command with environment setup for qsub
        ssh_command = [
            "sshpass", "-p", password,
            "ssh", f"{hpc_user}@{hpc_host}",
            f"bash -l -c 'cd {hpc_remote_path} && qsub {log_dir}/hpc_run.sh'"
        ]

        # Run the command
        result = subprocess.run(ssh_command, check=True, text=True, capture_output=True)

        # Extract the job ID from the result
        job_id = result.stdout.strip()
        messagebox.showinfo("Success", f"Job submitted successfully!\nJob ID: {job_id}")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Job submission failed.\n{e.stderr}")

    except Exception as e:
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {str(e)}")


def transfer_files_to_hpc(log_dir, remote_path, hpc_user, hpc_host, password):
    local_path = "."
    hpc_remote_path = "~/BTE-NO"

    # Use subprocess with `sshpass` for password-based file transfer
    import subprocess
    rsync_command = [
        "sshpass", "-p", password,
        "rsync", "-avz", "--progress",
        "--exclude-from", f"{local_path}/hpc_exclude.txt",
        f"{local_path}/src",
        f"{log_dir}/",
        f"{hpc_user}@{hpc_host}:{hpc_remote_path}"
    ]
    subprocess.run(rsync_command, check=True)


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
