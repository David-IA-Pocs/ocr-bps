import platform
import subprocess
import sys

print("=" * 55)
print("DIAGNÓSTICO DE GPU")
print("=" * 55)

print(f"Sistema operativo: {platform.platform()}")
print(f"Python: {sys.version}")

# 1. Detectar adaptadores gráficos de Windows
if platform.system() == "Windows":
    print("\n--- Adaptadores gráficos de Windows ---")
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, DriverVersion, Status | "
                "Format-Table -AutoSize"
            ],
            capture_output=True,
            text=True,
            timeout=20
        )
        print(result.stdout or result.stderr)
    except Exception as e:
        print(f"Error consultando Windows: {e}")

# 2. Consultar NVIDIA, si está instalada
print("\n--- NVIDIA ---")
try:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,"
            "memory.used,utilization.gpu",
            "--format=csv"
        ],
        capture_output=True,
        text=True,
        timeout=20
    )

    if result.returncode == 0:
        print(result.stdout)
    else:
        print("No se pudo consultar NVIDIA:")
        print(result.stderr or result.stdout)
except FileNotFoundError:
    print("nvidia-smi no está disponible en el PATH.")
except Exception as e:
    print(f"Error consultando NVIDIA: {e}")

# 3. Consultar proveedores de ONNX Runtime
print("\n--- ONNX Runtime ---")
try:
    import onnxruntime as ort

    print(f"Versión: {ort.__version__}")
    print("Proveedores disponibles:")
    for provider in ort.get_available_providers():
        print(f"  - {provider}")
except Exception as e:
    print(f"Error cargando ONNX Runtime: {e}")

print("\nDiagnóstico terminado.")