import subprocess

# Set GPU power limit to a specific wattage (e.g., 150W)
power_limit = 150  # Adjust wattage as needed
subprocess.run(["rocm-smi", "--setpower", str(power_limit)], check=True)