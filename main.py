import subprocess
from pathlib import Path


def run_hex_once() -> str:
    exe_path = Path(__file__).parent / "hex" / "hex.exe"

    result = subprocess.run(


        [str(exe_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def run_hex() -> str:
    # Path to the C executable you just ran in CLion
    exe_path = Path(__file__).parent / "hex" / "hex.exe"

    if not exe_path.exists():
        raise FileNotFoundError(f"hex.exe not found at {exe_path}")

    # Run the program and capture everything it prints
    result = subprocess.run(
        [str(exe_path)],
        capture_output=True,
        text=True,
        check=True
    )

    return result.stdout


if __name__ == "__main__":
    output = run_hex_once()
    print(output[:1000])  # just show the first 1000 characters
    #output = run_hex()
    #print("=== hex.exe output (from Python) ===")
    #print(output)

