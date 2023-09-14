# TEMPURA 🍤
Python Project Template 

## Directory Structure
```
.
├── Dockerfile
├── LICENSE
├── README.TEMPURA.md
├── README.md
├── compose.gpu.yml
├── compose.yml
├── gcp
│   ├── change_git_admin.sh
│   ├── install_docker.sh
│   └── reinstall_nvidia_driver.sh
├── notebooks
├── poetry.lock
├── pyproject.toml
├── src
│   └── __init__.py
└── tests
    └── __init__.py

```
## Requirements

### **Git**
- To clone the repository: 
  ```bash
  git clone git@github.com:osushinekotan/tempura.git
  ```
### **vscode**
- Installation:
    - Direct installer: Visual Studio Code
    - Using brew (for macOS):
        ```bash
        brew install visual-studio-code
        ```
- Extensions : 
    - `ms-vscode-remote.remote-containers`

### **docker** 
- Docker Desktop for Mac
- Docker Desktop for Windows
- For Google Compute Engine (or other platforms):
    ```bash
    sh gcp/install_docker.sh
    ```

## Features
- Utilizes `devcontainer` as the workspace
- Python version: `3.11` (modifiable)
- Package manager: `poetry (1.5.1)` (Python and poetry versions can be changed)
- Formatter: `black`
- Linter and Formatter: `ruff`
- Type checker: `mypy`


## GPU (GCE)
- If you are using a GCE instance with a GPU, set `compose.gpu.yml` as the `dockerComposeFile` in `.devcontainer/devcontainer.json`.
- Also, consider using `gcp/change_git_admin.sh` and `reinstall_nvidia_driver.sh` as needed.