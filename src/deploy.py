import subprocess
import sys
import os
import platform     # To handle platform-specific path separators for --add-data



# --- Configuration ---
APP_NAME = "AWC CV QC PyTorch"  # The desired name for your application executable
ENTRY_SCRIPT = "F:/JetBrains/PycharmProjects/AWC_CV_QC/src/inference.py" # The main Python script for your application
#MODEL_FILE = "runs/classify/11s/weights/best.onnx"
MODEL_FILE = "runs/classify/11s/weights/best.pt" # The path to your model file (relative to this script or absolute)
#FORMAT = "ONNX"
FORMAT = "PyTorch"



# --- PyInstaller Options ---
# Base command
command = [sys.executable, "-m", "PyInstaller", ENTRY_SCRIPT, "--name",
           APP_NAME, "--console", "--clean"]



# --- Platform-specific settings ---
# Ensure console is enabled for debugging output
data_separator = ";" if platform.system() == "Windows" else ":"



# --- Add Data Files ---
# Ensure the model file exists before adding it
if os.path.exists(MODEL_FILE):
    # Format: 'source_path{separator}destination_folder_in_bundle'
    # '.' means the root folder alongside the executable
    command.append(f"--add-data={MODEL_FILE}{data_separator}.")
else:
    print(f"Warning: Model file '{MODEL_FILE}' not found. Skipping --add-data for it.")
    # Decide if you want to exit if the model is missing
    # sys.exit(f"Error: Model file '{MODEL_FILE}' not found. Cannot proceed.")

# Add other assets if defined
# if 'OTHER_ASSETS' in locals():
#     for source, dest in OTHER_ASSETS:
#         if os.path.exists(source):
#             command.append(f"--add-data={source}{data_separator}{dest}")
#         else:
#             print(f"Warning: Asset '{source}' not found. Skipping --add-data for it.")



# --- Add Hidden Imports ---
# These are often needed for libraries like PyTorch, OpenCV, etc.
# Add more based on errors encountered during testing the built .exe
hidden_imports = [
    "ultralytics.engine.predictor",
    "ultralytics.engine.results",
    "torch",
    "torchvision", # Often needed with torch
    # Add other potential hidden imports below
    "numpy", # Often implicitly required
    "cv2",   # If your inference script uses OpenCV
    # "scipy",
    # "pandas",
    # "PIL", # Pillow
]

if FORMAT == "ONNX":
    hidden_imports.append("onnxruntime")
    print("Using ONNX Format.")
elif FORMAT == "PyTorch":
    print("Using PyTorch Format.")
else:
    sys.exit(1)

for imp in hidden_imports:
    command.append(f"--hidden-import={imp}")



# --- Add other PyInstaller options if needed ---
# !! Enable verbose runtime debugging !!
#command.append("--debug=all")
# command.append("--log-level=DEBUG") # Build-time debug logging (less critical now)

# command.append("--onefile") # Uncomment for single-file executable (debug with one-dir first)



# --- Execute the Command ---
print("Running PyInstaller with command (Runtime Debug Enabled):")
# Print the command in a more readable format
print(" ".join(f'"{arg}"' if " " in arg else arg for arg in command))
print("This will take a few minutes.")
print("-" * 30)

try:
    # Run the command
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    # Print stdout and stderr from the PyInstaller process
    print("PyInstaller STDOUT:")
    print(process.stdout)
    print("PyInstaller STDERR:")
    print(process.stderr) # Stderr might contain warnings even on success
    print("-" * 30)
    print(f"PyInstaller build completed successfully! Check the 'dist/{APP_NAME}' folder.")
    print("\n" + "="*40)
    print("IMPORTANT: The executable now includes runtime debug messages.")
    print("="*40 + "\n")

except FileNotFoundError:
    print("Error: PyInstaller command not found.")
    print("Make sure PyInstaller is installed in your Python environment:")
    print(f"  pip install pyinstaller")
    sys.exit(2)

except subprocess.CalledProcessError as e:
    # This catches errors during the PyInstaller execution
    print(f"Error: PyInstaller failed with exit code {e.returncode}")
    print("PyInstaller STDOUT:")
    print(e.stdout)
    print("PyInstaller STDERR:")
    print(e.stderr)
    print("-" * 30)
    print("Build failed. Check the errors above, especially for missing imports or data files.")
    sys.exit(3)

except Exception as e:
    # Catch any other unexpected errors
    print(f"An unexpected error occurred: {e}")
    sys.exit(4)
