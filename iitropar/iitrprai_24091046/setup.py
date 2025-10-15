# setup.py
import os
import sys
import time
import subprocess
import webbrowser
import venv
import shutil

BACKEND_DIR = "backend"
FRONTEND_DIR = "frontend"
VENV_DIR = "venv"
FRONTEND_URL = "http://127.0.0.1:3000"

# ---------- Helpers ----------
def run_cmd(cmd, cwd=None, check=True):
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=check, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def print_step(msg): print(f"\n[→] {msg}")
def print_success(msg): print(f"[SUCCESS] {msg}")
def print_fail(msg): print(f"[FAILED] {msg}")
def print_warn(msg): print(f"[WARNING] {msg}")

# ---------- Checks ----------
def check_docker():
    return run_cmd("docker --version") is not None

def check_compose():
    return run_cmd("docker compose version") is not None or run_cmd("docker-compose --version") is not None

def check_node():
    return run_cmd("node -v") is not None and run_cmd("npm -v") is not None

def docker_ready():
    """Check if Docker engine is up and responding"""
    result = run_cmd("docker info")
    return result is not None

# ---------- Wait for tool ----------
def wait_for_tool(tool_name, check_fn, download_url):
    if check_fn():
        return True

    choice = input(f"{tool_name} not found. Do you want to install it? (y/n): ").strip().lower()
    if choice != "y":
        print_warn(f"Skipping {tool_name} installation, continuing...")
        return False

    print_step(f"Opening {tool_name} download page...")
    time.sleep(3)
    webbrowser.open(download_url)

    print_step(f"Waiting for {tool_name} installation to complete...")
    while not check_fn():
        print(f"Still waiting for {tool_name}... (check again in 30s)")
        time.sleep(30)

    print_success(f"{tool_name} installed successfully!")
    return True

# ---------- Docker path ----------
def run_docker():
    print_step("Running with Docker...")
    subprocess.call("docker compose up --build", shell=True)

# ---------- Local venv path ----------
def run_local():
    # Setup Python venv
    print_step("Creating Python virtual environment...")
    if os.path.exists(VENV_DIR):
        shutil.rmtree(VENV_DIR)
    venv.create(VENV_DIR, with_pip=True)
    pip_path = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "pip")
    python_path = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python")
    print_success("Virtual environment created")

    # Install backend dependencies
    print_step("Installing backend requirements...")
    if not run_cmd(f"{pip_path} install -r {os.path.join(BACKEND_DIR, 'requirements.txt')}"):
        print_fail("Backend dependencies installation failed")
        return

    # Check Node.js
    if not wait_for_tool("Node.js & npm", check_node, "https://nodejs.org/en/download/"):
        print_fail("Cannot proceed without Node.js/npm")
        return

    # Install frontend dependencies
    print_step("Installing frontend dependencies...")
    if not run_cmd("npm install", cwd=FRONTEND_DIR):
        print_fail("Frontend dependencies installation failed")
        return
    print_success("Frontend dependencies installed")

    # Start backend first
    print_step("Starting backend server...")
    backend_proc = subprocess.Popen([python_path, "-m", "uvicorn", "main:app",
                                     "--host", "0.0.0.0", "--port", "8000"])

    # Wait a bit for backend to be up
    time.sleep(3)

    # Start frontend
    print_step("Starting frontend server...")
    frontend_proc = subprocess.Popen("npm start", cwd=FRONTEND_DIR, shell=True)
    print_success("Frontend started")

    # Open browser
    time.sleep(3)
    print_step(f"Opening {FRONTEND_URL} in browser...")
    webbrowser.open(FRONTEND_URL)

    backend_proc.wait()
    frontend_proc.wait()

# ---------- Main ----------
def main():
    print("=== Stock Prediction Setup ===")

    # Docker path
    if check_docker() and check_compose():
        if docker_ready():
            print_success("Docker & Compose running → starting containers")
            run_docker()
        else:
            print_fail("Docker Desktop is installed but not running.")
            print("Please start Docker Desktop and wait until it is fully initialized.")
            input("Press Enter once Docker Desktop is running...")
            if docker_ready():
                run_docker()
            else:
                print_fail("Docker still not responding. Falling back to local setup.")
                run_local()
    else:
        if wait_for_tool("Docker Desktop", lambda: check_docker() and check_compose(),
                         "https://www.docker.com/products/docker-desktop/"):
            run_docker()
        else:
            print_step("Proceeding with Local venv + npm setup...")
            run_local()

if __name__ == "__main__":
    main()
