import subprocess
import sys
import os

def run_script_with_config(script_name, config_path="config_first.json"):
    print(f"Running {script_name} with config {config_path}...")
    # Use sys.executable to ensure the script is run with the same Python interpreter
    # that is executing this script.
    result = subprocess.run([sys.executable, script_name, config_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script_name}:\n{result.stderr}")
    result.check_returncode() # This will raise an exception if the script returned a non-zero exit code
    print(f"{script_name} completed successfully.")

if __name__ == "__main__":
    scripts = [
        "split_pages.py",
        "deskew.py",
        "detect_lines.py",
        "reconstruct_table_full.py",
        "crop_to_table.py"
    ]

    # Ensure we are in the correct directory if the script is run from elsewhere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    config_path = "config_first.json"

    for script in scripts:
        try:
            run_script_with_config(script, config_path)
        except subprocess.CalledProcessError as e:
            print(f"Failed to run {script}: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"Error: {script} not found. Make sure it's in the same directory as First_Run.py or its path is correct.")
            sys.exit(1)
    
    print("\nAll steps completed successfully!")
