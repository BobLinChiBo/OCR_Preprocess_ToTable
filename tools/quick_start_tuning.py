#!/usr/bin/env python3
"""
Quick Start Parameter Tuning Script

This interactive script guides you through the complete parameter tuning process,
providing prompts and assistance at each stage.

Usage:
    python tools/quick_start_tuning.py

Features:
- Interactive prompts between each stage
- Automatic directory management
- Progress tracking
- Resume capability
"""

import sys
import subprocess
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TuningState:
    """Track the current state of the tuning process."""
    
    def __init__(self):
        self.stages = {
            'setup': False,
            'page_splitting': False,
            'deskewing': False,
            'roi_detection': False,
            'line_detection': False,
            'final_pipeline': False
        }
        self.state_file = Path("tools/tuning_state.txt")
        self.load_state()
    
    def load_state(self):
        """Load saved state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    for line in f:
                        if '=' in line:
                            stage, status = line.strip().split('=')
                            if stage in self.stages:
                                self.stages[stage] = status.lower() == 'true'
            except Exception as e:
                print(f"Warning: Could not load state file: {e}")
    
    def save_state(self):
        """Save current state to file."""
        try:
            with open(self.state_file, 'w') as f:
                for stage, completed in self.stages.items():
                    f.write(f"{stage}={completed}\n")
        except Exception as e:
            print(f"Warning: Could not save state file: {e}")
    
    def mark_completed(self, stage):
        """Mark a stage as completed."""
        if stage in self.stages:
            self.stages[stage] = True
            self.save_state()
    
    def is_completed(self, stage):
        """Check if a stage is completed."""
        return self.stages.get(stage, False)
    
    def get_next_stage(self):
        """Get the next uncompleted stage."""
        stage_order = ['setup', 'page_splitting', 'deskewing', 'roi_detection', 'line_detection', 'final_pipeline']
        for stage in stage_order:
            if not self.is_completed(stage):
                return stage
        return None


def print_banner():
    """Print welcome banner."""
    print("🚀 OCR PARAMETER TUNING - QUICK START")
    print("=" * 50)
    print("This interactive script will guide you through the complete")
    print("parameter tuning process step by step.")
    print()


def prompt_user(message, options=None):
    """Prompt user for input with optional validation."""
    while True:
        response = input(f"{message} ").strip().lower()
        
        if options is None:
            return response
        
        if response in [opt.lower() for opt in options]:
            return response
        
        print(f"Please enter one of: {', '.join(options)}")


def run_script(script_name, description):
    """Run a tuning script and handle errors."""
    print(f"\n🔧 {description}")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False


def check_directory_has_files(directory, pattern="*"):
    """Check if directory exists and has files."""
    path = Path(directory)
    if not path.exists():
        return False
    
    files = list(path.glob(pattern))
    return len(files) > 0


def guide_file_copying(source_stage, target_dir, file_pattern="*.jpg"):
    """Guide user through file copying process."""
    print(f"\n📁 FILE COPYING STEP")
    print("-" * 30)
    print(f"1. Open the results directory: data/output/tuning/{source_stage}/")
    print("2. Browse through the parameter combination folders")
    print("3. Visually evaluate which combination works best")
    print("4. Copy the best results to the next stage")
    print()
    
    # Check if target directory exists
    target_path = Path(target_dir)
    if not target_path.exists():
        print(f"Creating target directory: {target_dir}")
        target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Copy command example (replace 'best_folder' with actual folder name):")
    
    # Provide OS-specific copy commands
    if sys.platform.startswith('win'):
        print(f'copy "data\\output\\tuning\\{source_stage}\\best_folder\\{file_pattern}" "{target_dir}\\"')
    else:
        print(f"cp data/output/tuning/{source_stage}/best_folder/{file_pattern} {target_dir}/")
    
    print()
    
    # Wait for user to complete copying
    while True:
        response = prompt_user("Have you copied the best results to the target directory? (y/n)", ["y", "n"])
        
        if response == "y":
            # Verify files were copied
            if check_directory_has_files(target_dir, file_pattern):
                print("✅ Files found in target directory!")
                return True
            else:
                print("❌ No files found in target directory. Please check your copy command.")
                continue
        else:
            print("Please copy the files before continuing.")


def show_evaluation_tips(stage):
    """Show stage-specific evaluation tips."""
    tips = {
        'page_splitting': [
            "Look for clean separation at the book spine/gutter",
            "Check that no content is cut off from either page",
            "Verify consistent splitting across all test images"
        ],
        'deskewing': [
            "Check that text lines are horizontal", 
            "Verify table borders are straight",
            "Ensure no over-correction of already straight images",
            "Look at the angle analysis files for correction statistics"
        ],
        'roi_detection': [
            "Check that ROI crops focus on table content",
            "Verify headers/footers are appropriately removed",
            "Ensure no important content is cut off",
            "Review area ratio statistics in analysis files"
        ],
        'line_detection': [
            "Look at line visualization images (red=horizontal, green=vertical)",
            "Check that major table lines are detected",
            "Verify minimal noise/false detections",
            "Review detection rate statistics"
        ]
    }
    
    print(f"\n💡 EVALUATION TIPS FOR {stage.upper().replace('_', ' ')}:")
    print("-" * 40)
    for tip in tips.get(stage, []):
        print(f"   • {tip}")
    print()


def main():
    """Main interactive tuning process."""
    print_banner()
    
    # Initialize state tracking
    state = TuningState()
    
    # Check if we're resuming
    next_stage = state.get_next_stage()
    if next_stage != 'setup':
        print(f"🔄 Resuming from: {next_stage.replace('_', ' ').title()}")
        resume = prompt_user("Continue from where you left off? (y/n)", ["y", "n"])
        if resume != "y":
            # Reset state
            state = TuningState()
            state.save_state()
            next_stage = 'setup'
    
    print("📋 This process will guide you through:")
    print("   1. Setup and validation")
    print("   2. Page splitting parameter tuning")
    print("   3. Deskewing parameter tuning")
    print("   4. ROI detection parameter tuning")
    print("   5. Line detection parameter tuning")
    print("   6. Final tuned pipeline execution")
    print()
    
    ready = prompt_user("Ready to start? (y/n)", ["y", "n"])
    if ready != "y":
        print("Exiting. Run this script again when you're ready!")
        return
    
    # Stage 1: Setup
    if not state.is_completed('setup'):
        print("\n" + "="*50)
        print("STAGE 1: SETUP AND VALIDATION")
        print("="*50)
        
        success = run_script("tools/setup_tuning.py", "Setup and validation")
        if success:
            state.mark_completed('setup')
        else:
            print("❌ Setup failed. Please fix issues before continuing.")
            return
    
    # Stage 2: Page Splitting
    if not state.is_completed('page_splitting'):
        print("\n" + "="*50)
        print("STAGE 2: PAGE SPLITTING PARAMETER TUNING")
        print("="*50)
        
        success = run_script("tools/tune_page_splitting.py", "Page splitting parameter tuning")
        if success:
            show_evaluation_tips('page_splitting')
            
            if guide_file_copying("01_split_pages", "data/output/tuning/02_deskewed_input"):
                state.mark_completed('page_splitting')
            else:
                print("❌ Please complete file copying before continuing.")
                return
        else:
            print("❌ Page splitting tuning failed.")
            return
    
    # Stage 3: Deskewing
    if not state.is_completed('deskewing'):
        print("\n" + "="*50)
        print("STAGE 3: DESKEWING PARAMETER TUNING")
        print("="*50)
        
        success = run_script("tools/tune_deskewing.py", "Deskewing parameter tuning")
        if success:
            show_evaluation_tips('deskewing')
            
            if guide_file_copying("02_deskewed", "data/output/tuning/03_roi_input"):
                state.mark_completed('deskewing')
            else:
                print("❌ Please complete file copying before continuing.")
                return
        else:
            print("❌ Deskewing tuning failed.")
            return
    
    # Stage 4: ROI Detection
    if not state.is_completed('roi_detection'):
        print("\n" + "="*50)
        print("STAGE 4: ROI DETECTION PARAMETER TUNING")
        print("="*50)
        
        success = run_script("tools/tune_roi_detection.py", "ROI detection parameter tuning")
        if success:
            show_evaluation_tips('roi_detection')
            
            if guide_file_copying("03_roi_detection", "data/output/tuning/04_line_input", "*_roi.jpg"):
                state.mark_completed('roi_detection')
            else:
                print("❌ Please complete file copying before continuing.")
                return
        else:
            print("❌ ROI detection tuning failed.")
            return
    
    # Stage 5: Line Detection
    if not state.is_completed('line_detection'):
        print("\n" + "="*50)
        print("STAGE 5: LINE DETECTION PARAMETER TUNING")
        print("="*50)
        
        success = run_script("tools/tune_line_detection.py", "Line detection parameter tuning")
        if success:
            show_evaluation_tips('line_detection')
            
            print("\n📝 PARAMETER RECORDING:")
            print("1. Note the best parameters from each stage")
            print("2. Update TUNED_PARAMETERS in tools/run_tuned_pipeline.py")
            print("3. Save your optimal parameters for future reference")
            
            updated = prompt_user("Have you updated the parameters in run_tuned_pipeline.py? (y/n)", ["y", "n"])
            if updated == "y":
                state.mark_completed('line_detection')
            else:
                print("❌ Please update the parameters before proceeding to the final stage.")
                return
        else:
            print("❌ Line detection tuning failed.")
            return
    
    # Stage 6: Final Pipeline
    if not state.is_completed('final_pipeline'):
        print("\n" + "="*50)
        print("STAGE 6: FINAL TUNED PIPELINE")
        print("="*50)
        
        print("Running the complete pipeline with your tuned parameters...")
        
        success = run_script("tools/run_tuned_pipeline.py data/input/test_images/ --verbose", 
                           "Final tuned pipeline execution")
        if success:
            state.mark_completed('final_pipeline')
        else:
            print("❌ Final pipeline failed.")
            return
    
    # Completion
    print("\n" + "🎉"*50)
    print("PARAMETER TUNING COMPLETED SUCCESSFULLY!")
    print("🎉"*50)
    print()
    print("✅ All stages completed successfully")
    print("📊 Your optimized parameters are now ready for production use")
    print("📁 Final results are in: data/output/tuned_pipeline/")
    print()
    print("💡 NEXT STEPS:")
    print("   • Compare results with the default pipeline")
    print("   • Test on a larger dataset")
    print("   • Document your optimal parameters")
    print("   • Consider version controlling your parameter set")
    print()
    print("📝 Clean up: Remove tuning_state.txt if you want to start fresh next time")
    
    # Clean up state file
    cleanup = prompt_user("Remove state tracking file? (y/n)", ["y", "n"])
    if cleanup == "y":
        try:
            state.state_file.unlink()
            print("✅ State file removed")
        except:
            print("❌ Could not remove state file")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("Run the script again to resume from where you left off.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Check the error details and try again.")