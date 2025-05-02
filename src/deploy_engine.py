"""
Not Ready.

# --- !! IMPORTANT TENSORRT NOTE !! ---
# TensorRT executables created with PyInstaller have a major dependency:
# The target machine MUST have the correct NVIDIA drivers, CUDA Toolkit,
# and TensorRT runtime libraries installed system-wide that are compatible
# with the engine file and the TensorRT Python library version used.
# PyInstaller bundles the Python code and the .engine file, NOT the core
# TensorRT runtime itself.
"""



import subprocess
import sys
import os
import platform     # To handle platform-specific path separators for --add-data



# --- Configuration ---
APP_NAME = "AWC CV QC TensorRT"  # The desired name for your application executable
ENTRY_SCRIPT = "F:/JetBrains/PycharmProjects/AWC_CV_QC/src/inference.py" # The main Python script for your application
MODEL_FILE = "runs/classify/11s/weights/best.engine" # The path to your model file (relative to this script or absolute)
# OTHER_ASSETS = [("path/to/your/font.ttf", ".")] # Example: (source, destination_in_bundle)



# --- PyInstaller Options ---
# Base command
command = [
    sys.executable,  # Use the current Python interpreter to run PyInstaller module
    "-m", "PyInstaller",
    ENTRY_SCRIPT,
    "--name", APP_NAME,
]

# --- Platform-specific settings ---
# Build as a windowed app (no console) by default.
# Comment out '--windowed'/'--noconsole' and uncomment '--console' for debugging.
data_separator = ";" if platform.system() == "Windows" else ":"
if platform.system() == "Windows":
    # Use '--windowed' for no console on Windows
    command.append("--windowed")
    # command.append("--console") # Uncomment for debugging console
else:
    # Use '--noconsole' for no console on macOS/Linux (if desired)
    command.append("--noconsole")
    # command.append("--console") # Uncomment for debugging console

# --- Add Data Files ---
# Ensure the TensorRT engine file exists before adding it
if os.path.exists(MODEL_FILE):
    # Format: 'source_path{separator}destination_folder_in_bundle'
    # '.' means the root folder alongside the executable
    command.append(f"--add-data={MODEL_FILE}{data_separator}.")
else:
    print(f"Warning: TensorRT Engine file '{MODEL_FILE}' not found. Skipping --add-data for it.")
    # Decide if you want to exit if the model is missing
    sys.exit(f"Error: TensorRT Engine file '{MODEL_FILE}' not found. Cannot proceed.")

# Add other assets if defined
# if 'OTHER_ASSETS' in locals():
#     for source, dest in OTHER_ASSETS:
#         if os.path.exists(source):
#             command.append(f"--add-data={source}{data_separator}{dest}")
#         else:
#             print(f"Warning: Asset '{source}' not found. Skipping --add-data for it.")


# --- Add Hidden Imports ---
# Add imports potentially missed by PyInstaller analysis.
# Crucial for TensorRT and associated libraries.
hidden_imports = [
    "tensorrt",   # Essential for TensorRT Python API
    # "pycuda",     # Add if your script uses PyCUDA directly
    "numpy",      # Very likely needed
    "cv2",        # Likely needed for image pre/post-processing
    # --- Add other potential hidden imports below ---
    # "scipy",
    # "pandas",
    # "PIL", # Pillow
    # --- Removed imports not typically needed for pure TensorRT inference ---
    # "onnxruntime",
    # "ultralytics.engine.predictor", # Unless your TRT script still uses parts of it
    # "ultralytics.engine.results",   # Unless your TRT script still uses parts of it
    # "torch",                        # Unless your TRT script still uses parts of it
    # "torchvision",                  # Unless your TRT script still uses parts of it
]
for imp in hidden_imports:
    command.append(f"--hidden-import={imp}")

# --- Add other PyInstaller options if needed ---
# Note on Multiprocessing: PyInstaller uses multiprocessing internally for tasks like
# compression, especially with --onefile. There isn't a specific flag to control
# the number of cores for the entire analysis process, but it will generally
# utilize available resources where applicable.
# num_cores = multiprocessing.cpu_count()
# print(f"System has {num_cores} CPU cores available.")

command.append("--clean") # Clean PyInstaller cache and remove temporary files before building
command.append("--log-level=INFO") # Set log level (DEBUG, INFO, WARN, ERROR, CRITICAL)

# command.append("--onefile") # Uncomment for single-file executable (debug with one-dir first)


# --- Execute the Command ---
print(f"Running PyInstaller for TensorRT script '{ENTRY_SCRIPT}' with command:")
# Print the command in a more readable format
print(" ".join(f'"{arg}"' if " " in arg else arg for arg in command))
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
    print("REMINDER: The target machine MUST have compatible NVIDIA drivers,")
    print("CUDA Toolkit, and TensorRT runtime libraries installed for this")
    print(f"executable ('dist/{APP_NAME}/{APP_NAME}.exe') to function correctly.")
    print("="*40 + "\n")


except FileNotFoundError:
    print("Error: PyInstaller command not found.")
    print("Make sure PyInstaller is installed in your Python environment:")
    print(f"  pip install pyinstaller")
    sys.exit(1)

except subprocess.CalledProcessError as e:
    # This catches errors during the PyInstaller execution
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
    sys.exit(1)
