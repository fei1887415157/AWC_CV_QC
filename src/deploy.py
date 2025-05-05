"""
!!! NOT WORKING !!!

Distribution and Deployment Instructions:
Go to /src/dist,
copy and paste the "AWC CV QC PyTorch" folder.
"""

import shutil
import subprocess
import sys
import os
import platform # To handle platform-specific path separators for --add-data
import inference



# --- Configuration ---
APP_NAME = "AWC CV QC PyTorch"  # The desired name for your application executable
ENTRY_SCRIPT = "inference.py" # The main Python script for your application
MODEL_FILE = "runs/classify/11s/weights/best.pt" # The path to your model file (relative to this script or absolute)
inference.set_model_path("_internal/best.pt") # Change model path for inference, since it will be at this location when packaged



# --- PyInstaller Options ---
# Base command
# "--onefile" slower and not working?
# "--debug=all" must use or will not work
command = [sys.executable, "-m", "PyInstaller", ENTRY_SCRIPT, "--name", APP_NAME, "--debug=all"]



# --- Platform-specific settings ---
# Ensure console is enabled for debugging output
data_separator = ";" if platform.system() == "Windows" else ":"
# if platform.system() == "Windows":
    # Use '--windowed' for no console on Windows
    #command.append("--windowed") # Keep commented out for debugging
    # data_separator = ";"
# else:
    # Use '--noconsole' for no console on macOS/Linux (if desired)
    # command.append("--noconsole") # Keep commented out for debugging
    # data_separator = ":"



# --- Add Data Files ---
# Ensure the model file exists before adding it
if os.path.exists(MODEL_FILE):
    # Format: 'source_path{separator}destination_folder_in_bundle'
    # '.' means the root folder alongside the executable
    command.append(f"--add-data={MODEL_FILE}{data_separator}.")
else:
    sys.exit(f"Error: Model file '{MODEL_FILE}' not found. Cannot proceed.")



# --- Add Hidden Imports ---
# These are often needed for libraries like PyTorch, OpenCV, etc.
# Add more based on errors encountered during testing the built .exe
hidden_imports = [
    "ultralytics.engine.predictor",
    "ultralytics.engine.results",
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "optree"
]
for imp in hidden_imports:
    command.append(f"--hidden-import={imp}")



# --- Clear the dist folder to prevent conflicts ---
if os.path.isdir("dist\\" + APP_NAME):
    print("Clearing dist folder")
    shutil.rmtree("dist\\" + APP_NAME)
    print("Cleared dist folder")



# --- Execute the Command ---
# Print the command in a more readable format
print(" ".join(f'"{arg}"' if " " in arg else arg for arg in command))
print("This will take about 3 minutes.")
print("-" * 30)

try:
    # Run the command
    process = subprocess.run(command, check=True, capture_output=True, text=True)

except FileNotFoundError:
    sys.exit("Error: PyInstaller not found.")

except subprocess.CalledProcessError as e:
    print(f"Error: PyInstaller failed with exit code {e.returncode}")
    print("PyInstaller STDOUT:")
    print(e.stdout)
    print("PyInstaller STDERR:")
    print(e.stderr)
    print("-" * 30)
    print("Build failed. Check the errors above, especially for missing imports or data files.")
    sys.exit(1)

except Exception as e:
    # Catch any other unexpected errors
    print(f"An unexpected error occurred: {e}")
    sys.exit(10)