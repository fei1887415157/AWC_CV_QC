import subprocess
import sys
import os
import platform # To handle platform-specific path separators for --add-data



# --- Configuration ---
APP_NAME = "AWC CV QC"  # The desired name for your application executable
ENTRY_SCRIPT = "F:/JetBrains/PycharmProjects/AWC_CV_QC/src/inference.py" # The main Python script for your application
MODEL_FILE = "runs/classify/11s/weights/best.pt" # The path to your model file (relative to this script or absolute)
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
if platform.system() == "Windows":
    # Use '--windowed' for no console on Windows
    command.append("--windowed")
    # Data path separator for Windows
    data_separator = ";"
else:
    # Use '--noconsole' for no console on macOS/Linux (if desired)
    # command.append("--noconsole") # Uncomment if you don't want a console on Mac/Linux
    # Data path separator for macOS/Linux
    data_separator = ":"

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
    #"builtins",
    #"ultralytics.engine.predictor",
    #"ultralytics.engine.results",
    #"torch",
    #"torchvision", # Often needed with torch
    #"cv2",
    # Add other potential hidden imports below
    # "numpy",
    # "scipy",
    # "pandas",
    # "PIL", # Pillow
]
for imp in hidden_imports:
    command.append(f"--hidden-import={imp}")



# --- Add other PyInstaller options if needed ---
# command.append("--onefile") # Uncomment for single-file executable (debug with one-dir first)
# command.append("--clean") # Clean PyInstaller cache and remove temporary files before building
# command.append("--log-level=DEBUG") # For more verbose output during build



# --- Execute the Command ---
print("Running PyInstaller with command:")
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
