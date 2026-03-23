"""
install_deps.py
Creates a virtual environment named .venv (if missing) and installs packages from
requirements.txt using the venv's Python interpreter. Run this script from the
workspace root (where requirements.txt lives):

    python install_deps.py

This is a cross-platform helper. On Windows it will create .venv\Scripts\python.exe
and use that to install requirements.
"""
import os
import subprocess
import sys
import venv

VENV_DIR = ".venv"

def create_venv(path: str):
    if not os.path.exists(path):
        print(f"Creating virtual environment at {path}...")
        venv.create(path, with_pip=True)
    else:
        print(f"Virtual environment already exists at {path}")

def get_venv_python(path: str) -> str:
    if os.name == 'nt':
        return os.path.join(path, 'Scripts', 'python.exe')
    else:
        return os.path.join(path, 'bin', 'python')

def run_install(python_exe: str):
    print(f"Using python: {python_exe}")
    subprocess.check_call([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([python_exe, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def main():
    try:
        create_venv(VENV_DIR)
        py = get_venv_python(VENV_DIR)
        if not os.path.exists(py):
            raise FileNotFoundError(f"Python executable not found in venv: {py}")
        run_install(py)
        print('\nDependencies installed successfully.')
        print('Activate the venv with:')
        if os.name == 'nt':
            print('    .\\.venv\\Scripts\\Activate.ps1  (PowerShell)')
            print('    .\\.venv\\Scripts\\activate.bat  (cmd.exe)')
        else:
            print('    source .venv/bin/activate')
    except subprocess.CalledProcessError as e:
        print('Installation failed with a subprocess error:')
        print(e)
        sys.exit(1)
    except Exception as exc:
        print('Error:', exc)
        sys.exit(1)

if __name__ == '__main__':
    main()
