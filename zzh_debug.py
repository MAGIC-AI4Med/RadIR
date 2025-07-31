import subprocess
import json

# 获取所有 Conda 环境
def get_conda_envs():
    result = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True)
    envs = json.loads(result.stdout)["envs"]
    return envs

# 获取指定环境的 Python 和 PyTorch 版本
def get_package_versions(env_path):
    python_version = "N/A"
    torch_version = "N/A"

    python_executable = f"{env_path}/bin/python"

    try:
        result = subprocess.run(
            [python_executable, "-c", "import sys; print(sys.version.split()[0])"],
            capture_output=True, text=True
        )
        python_version = result.stdout.strip() if result.returncode == 0 else "N/A"

        result = subprocess.run(
            [python_executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True
        )
        torch_version = result.stdout.strip() if result.returncode == 0 else "N/A"
    except Exception:
        pass  # 忽略异常（可能环境中没有安装 torch）

    return python_version, torch_version

# 遍历所有环境并获取版本信息
envs = get_conda_envs()

print(f"{'Environment':<20} {'Python Version':<15} {'PyTorch Version'}")
print("=" * 50)

for env in envs:
    env_name = env.split("/")[-1]  # 提取环境名称
    python_ver, torch_ver = get_package_versions(env)
    print(f"{env_name:<20} {python_ver:<15} {torch_ver}")