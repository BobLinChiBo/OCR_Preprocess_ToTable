"""Unified OCR pipeline with all processing modes."""

import argparse
import json
import time
import traceback
import sys
import gc
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

from .config import (
    Config,
    get_default_config,
    Stage1Config,
    Stage2Config,
    get_stage1_config,
    get_stage2_config,
)
from .processors import (
    load_image,
    save_image,
    get_image_files,
    split_two_page_image,
    remove_margin_inscribed,
    deskew_image,
    detect_table_lines,
    remove_marks,
    remove_tags,
    create_table_lines_mask,
    visualize_detected_lines,
    detect_table_structure,
    visualize_table_structure,
    crop_to_table_borders,
    table_recovery,
    cut_vertical_strips,
    binarize_image,
)
from .stage1_processor import Stage1Processor
from .stage2_processor import Stage2Processor


# Module-level worker functions for multiprocessing
def _process_stage1_with_disk_save(
    image_path: Path,
    stage1_config: Stage1Config,
    debug_dir: Optional[Path] = None
) -> Tuple[Path, Optional[List[Path]], Optional[str], float, bool]:
    """Worker function for processing Stage 1 with disk saves.
    
    Args:
        image_path: Path to the image
        stage1_config: Stage 1 configuration
        debug_dir: Optional debug directory
        
    Returns:
        Tuple of (image_path, cropped_paths, error_message, elapsed_time, is_memory_error)
    """
    start_time = time.time()
    try:
        # Create processor
        processor = Stage1Processor(stage1_config)
        
        # Process with disk saves
        results = processor.process_image(
            image_path,
            memory_mode=False,  # Force disk saves
            debug_dir=debug_dir
        )
        
        # Extract cropped paths
        cropped_paths = [result['cropped_path'] for result in results]
        
        # Clean up memory
        del results
        gc.collect()
        
        elapsed = time.time() - start_time
        return (image_path, cropped_paths, None, elapsed, False)
        
    except cv2.error as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        # Check if it's a memory allocation error
        is_memory_error = "Insufficient memory" in error_msg or "Failed to allocate" in error_msg
        if is_memory_error:
            error_msg = f"Memory error processing {image_path.name}: {error_msg}"
        else:
            error_msg = f"OpenCV error processing {image_path}: {error_msg}"
        return (image_path, None, error_msg, elapsed, is_memory_error)
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error processing {image_path}: {str(e)}\n{traceback.format_exc()}"
        return (image_path, None, error_msg, elapsed, False)


def _process_stage2_worker(
    image_path: Path,
    stage2_config: Stage2Config,
    debug_dir: Optional[Path] = None
) -> Tuple[Path, Optional[Dict[str, Any]], Optional[str], float]:
    """Worker function for processing a single cropped table through Stage 2.
    
    This function must be at module level for multiprocessing to work on Windows.
    
    Args:
        image_path: Path to the cropped table image
        stage2_config: Stage 2 configuration
        debug_dir: Optional debug directory
        
    Returns:
        Tuple of (image_path, results_dict, error_message, elapsed_time)
    """
    start_time = time.time()
    try:
        # Create processor
        processor = Stage2Processor(stage2_config)
        
        # Load the cropped table image
        table_image = load_image(image_path)
        base_name = image_path.stem.replace("_border_cropped", "")
        
        # Process through Stage 2
        results = processor.process_cropped_table(
            table_image,
            base_name,
            memory_mode=stage2_config.memory_mode,
            debug_dir=debug_dir
        )
        
        # Clean up memory
        del table_image
        gc.collect()
        
        elapsed = time.time() - start_time
        return (image_path, results, None, elapsed)
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error processing {image_path}: {str(e)}\n{traceback.format_exc()}"
        gc.collect()  # Clean up even on error
        return (image_path, None, error_msg, elapsed)


def _process_single_image_worker(
    image_path: Path,
    stage1_config: Stage1Config,
    stage2_config: Stage2Config,
    stage1_memory_mode: bool = True,
    stage2_memory_mode: bool = True
) -> Tuple[Path, Optional[List[Path]], Optional[str], float]:
    """Worker function for processing a single image in a worker process.
    
    This function must be at module level for multiprocessing to work on Windows.
    
    Args:
        image_path: Path to the image
        stage1_config: Stage 1 configuration
        stage2_config: Stage 2 configuration
        stage1_memory_mode: Whether to use memory mode for Stage 1
        stage2_memory_mode: Whether to use memory mode for Stage 2
        
    Returns:
        Tuple of (image_path, output_paths, error_message, elapsed_time)
    """
    start_time = time.time()
    try:
        # Create processors
        stage1_processor = Stage1Processor(stage1_config)
        stage2_processor = Stage2Processor(stage2_config)
        
        # Process Stage 1
        stage1_results = stage1_processor.process_image(
            image_path,
            memory_mode=stage1_memory_mode,
            debug_dir=None
        )
        
        output_paths = []
        
        # Process Stage 2 for each result
        for result in stage1_results:
            stage2_result = stage2_processor.process_cropped_table(
                result['image'],
                result['page_name'].replace('_border_cropped', ''),
                memory_mode=stage2_memory_mode,
                debug_dir=None
            )
            output_paths.extend(stage2_result.get('final_outputs', []))
        
        # Clean up memory
        del stage1_results
        gc.collect()
        
        elapsed = time.time() - start_time
        return (image_path, output_paths, None, elapsed)
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error processing {image_path}: {str(e)}\n{traceback.format_exc()}"
        gc.collect()  # Clean up even on error
        return (image_path, None, error_msg, elapsed)


class OCRPipeline:
    """Simple OCR table extraction pipeline (legacy compatibility)."""

    def __init__(self, config: Config = None):
        """Initialize pipeline with configuration."""
        self.config = config or get_default_config()
        self.config.create_output_dirs()

    def process_image(self, image_path: Path) -> List[Path]:
        """Process a single image through the pipeline."""
        if self.config.verbose:
            print(f"Processing: {image_path}")

        # Load image
        image = load_image(image_path)

        # Split into two pages
        right_page, left_page = split_two_page_image(
            image,
            search_ratio=self.config.search_ratio,
            line_len_frac=getattr(self.config, 'line_len_frac', 0.3),
            line_thick=getattr(self.config, 'line_thick', 3),
            peak_thr=getattr(self.config, 'peak_thr', 0.3)
        )

        output_paths = []

        # Process each page
        for page_name, page in [("left", left_page), ("right", right_page)]:
            # Margin removal (preprocessing step)
            processing_image = page
            if self.config.enable_margin_removal:
                processing_image = remove_margin_inscribed(
                    page,
                    blur_ksize=getattr(self.config, 'inscribed_blur_ksize', 5),
                    close_ksize=getattr(self.config, 'inscribed_close_ksize', 25),
                    close_iter=getattr(self.config, 'inscribed_close_iter', 2),
                )

                if self.config.verbose:
                    print(f"    Margin removed: {processing_image.shape} (from {page.shape})")

            # Deskew
            deskewed, _ = deskew_image(
                processing_image,
                coarse_range=self.config.coarse_range,
                coarse_step=self.config.coarse_step,
                fine_range=self.config.fine_range,
                fine_step=self.config.fine_step,
                min_angle_correction=self.config.min_angle_correction,
            )

            # Detect table lines
            h_lines, v_lines = detect_table_lines(
                deskewed,
                threshold=self.config.threshold,
                horizontal_kernel_size=self.config.horizontal_kernel_size,
                vertical_kernel_size=self.config.vertical_kernel_size,
                alignment_threshold=self.config.alignment_threshold,
                h_min_length_image_ratio=self.config.h_min_length_image_ratio,
                h_min_length_relative_ratio=self.config.h_min_length_relative_ratio,
                v_min_length_image_ratio=self.config.v_min_length_image_ratio,
                v_min_length_relative_ratio=self.config.v_min_length_relative_ratio,
                min_aspect_ratio=self.config.min_aspect_ratio,
                max_h_length_ratio=self.config.max_h_length_ratio,
                max_v_length_ratio=self.config.max_v_length_ratio,
                close_line_distance=self.config.close_line_distance,
                skew_tolerance=getattr(self.config, 'skew_tolerance', 0),
                skew_angle_step=getattr(self.config, 'skew_angle_step', 0.2),
            )

            # Use deskewed image directly (table cropping is done in Stage 1)
            cropped = deskewed

            # Save result
            output_name = f"{image_path.stem}_{page_name}.jpg"
            output_path = self.config.output_dir / output_name
            save_image(cropped, output_path)
            output_paths.append(output_path)

            if self.config.verbose:
                print(f"  Saved: {output_path}")

        return output_paths

    def process_directory(self, input_dir: Path = None) -> List[Path]:
        """Process all images in input directory."""
        input_dir = input_dir or self.config.input_dir

        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        image_files = get_image_files(input_dir)

        if not image_files:
            print(f"No image files found in: {input_dir}")
            return []

        print(f"Found {len(image_files)} images to process")

        all_outputs = []
        for image_path in image_files:
            try:
                outputs = self.process_image(image_path)
                all_outputs.extend(outputs)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        print("\n" + "-"*50)
        print(f"[PROCESSING COMPLETE] {len(all_outputs)} files created")
        print(f"Output location: {self.config.output_dir}")
        print("-"*50)
        return all_outputs


class TwoStageOCRPipeline:
    """Unified two-stage OCR pipeline with all processing modes."""

    def __init__(
        self, stage1_config: Stage1Config = None, stage2_config: Stage2Config = None
    ):
        """Initialize two-stage pipeline with configurations."""
        self.stage1_config = stage1_config or get_stage1_config()
        self.stage2_config = stage2_config or get_stage2_config()
        
        # Initialize unified stage processors
        self.stage1_processor = Stage1Processor(self.stage1_config)
        self.stage2_processor = Stage2Processor(self.stage2_config)
        
        # Generate single timestamp for entire pipeline run
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.debug_run_dir_s1 = None
        self.debug_run_dir_s2 = None
        self.debug_base_dir = None
        
        # Create debug directories if needed
        self._init_debug_dirs()
    
    def _init_debug_dirs(self):
        """Initialize debug directories if debug mode is enabled."""
        if ((self.stage1_config.save_debug_images and self.stage1_config.debug_dir) or 
            (self.stage2_config.save_debug_images and self.stage2_config.debug_dir)):
            # Use the first available debug_dir as the base
            base_debug_dir = self.stage1_config.debug_dir if self.stage1_config.debug_dir else self.stage2_config.debug_dir
            self.debug_base_dir = base_debug_dir / f"{self.run_timestamp}_run"
            self.debug_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create debug directories if debug mode is enabled
        if self.stage1_config.save_debug_images and self.stage1_config.debug_dir:
            self.debug_run_dir_s1 = self.debug_base_dir / "stage1"
            self.debug_run_dir_s1.mkdir(parents=True, exist_ok=True)
            
        if self.stage2_config.save_debug_images and self.stage2_config.debug_dir:
            self.debug_run_dir_s2 = self.debug_base_dir / "stage2"
            self.debug_run_dir_s2.mkdir(parents=True, exist_ok=True)
    
    def _save_run_info(self, stage: str, input_path: Path, num_images: int, status: str = "started"):
        """Save metadata about the pipeline run."""
        debug_dir = self.debug_run_dir_s1 if stage == "stage1" else self.debug_run_dir_s2
        if not debug_dir:
            return
            
        run_info = {
            "timestamp": self.run_timestamp,
            "stage": stage,
            "input_path": str(input_path),
            "num_images": num_images,
            "status": status,
            "config": {
                "save_debug_images": True,
                "debug_dir": str(self.debug_base_dir) if self.debug_base_dir else None,
                "verbose": self.stage1_config.verbose if stage == "stage1" else self.stage2_config.verbose
            }
        }
        
        info_file = debug_dir / "run_info.json"
        with open(info_file, "w") as f:
            json.dump(run_info, f, indent=2)

    def run_stage1(self, input_path: Path = None) -> List[Path]:
        """
        Run Stage 1: Initial processing and table cropping.

        Args:
            input_path: Input directory or single image file (optional)

        Returns:
            List of cropped table image paths ready for Stage 2
        """
        if self.stage1_config.verbose:
            if self.stage1_config.save_debug_images:
                print("*** STARTING STAGE 1: INITIAL PROCESSING (DEBUG MODE) ***")
                print("=" * 58)
                print(f"Debug images will be saved to: {self.debug_run_dir_s1}")
            else:
                print("*** STARTING STAGE 1: INITIAL PROCESSING ***")
                print("=" * 50)

        # Process images through complete Stage 1 pipeline
        input_path = input_path or self.stage1_config.input_dir
        cropped_tables = []

        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        # Handle single file or directory
        if input_path.is_file():
            image_files = [input_path]
        else:
            image_files = get_image_files(input_path)
            if not image_files:
                raise ValueError(f"No image files found in: {input_path}")

        if self.stage1_config.verbose:
            print(f"Found {len(image_files)} images to process")

        # Save run info at start
        self._save_run_info("stage1", input_path, len(image_files), "started")

        # Process each image using the unified Stage1Processor
        for image_path in image_files:
            try:
                # Use the Stage1Processor for all processing
                results = self.stage1_processor.process_image(
                    image_path,
                    memory_mode=self.stage1_config.memory_mode,
                    debug_dir=self.debug_run_dir_s1
                )
                
                # Collect cropped table paths from results
                for result in results:
                    cropped_tables.append(result['cropped_path'])
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if self.stage1_config.verbose:
            print(
                f"\n*** STAGE 1 COMPLETE: {len(cropped_tables)} cropped tables ready for Stage 2 ***"
            )
            print(f"Output: {self.stage1_config.output_dir / '07_border_cropped'}")
            if self.stage1_config.save_debug_images and self.debug_run_dir_s1:
                print(f"Debug images saved to: {self.debug_run_dir_s1}")

        # Save run info at completion
        self._save_run_info("stage1", input_path, len(image_files), "completed")

        return cropped_tables

    def run_stage2(self, input_dir: Path = None) -> List[Path]:
        """
        Run Stage 2: Refinement processing on cropped tables.

        Args:
            input_dir: Directory with cropped tables from Stage 1 (optional)

        Returns:
            List of final refined table image paths
        """
        if self.stage2_config.verbose:
            if self.stage2_config.save_debug_images:
                print("\n*** STARTING STAGE 2: REFINEMENT PROCESSING (DEBUG MODE) ***")
                print("=" * 61)
                print(f"Debug images will be saved to: {self.debug_run_dir_s2}")
            else:
                print("\n*** STARTING STAGE 2: REFINEMENT PROCESSING ***")
                print("=" * 50)

        input_dir = input_dir or self.stage2_config.input_dir

        if not input_dir.exists():
            raise ValueError(
                f"Stage 1 output not found: {input_dir}. Run Stage 1 first."
            )

        image_files = get_image_files(input_dir)
        if not image_files:
            raise ValueError(f"No cropped table images found in: {input_dir}")

        if self.stage2_config.verbose:
            print(f"Found {len(image_files)} cropped tables from Stage 1")

        # Save run info at start
        self._save_run_info("stage2", input_dir, len(image_files), "started")

        # Check if parallel processing is enabled
        if self.stage2_config.parallel_processing:
            if self.stage2_config.verbose:
                print(f"Using parallel processing (workers: {self.stage2_config.max_workers or max(1, cpu_count() - 1)})")
            refined_tables = self._run_stage2_parallel(
                image_files,
                max_workers=self.stage2_config.max_workers,
                batch_size=self.stage2_config.batch_size,
                show_progress=self.stage2_config.verbose
            )
        else:
            # Sequential processing (existing code)
            refined_tables = []
            output_locations = {
                "04_table_recovered": [],
                "05_vertical_strips": [],
                "06_binarized": [],
                "07_final_deskewed": []
            }

            # Process each cropped table using the unified Stage2Processor
            for image_path in image_files:
                try:
                    # Load cropped table
                    table_image = load_image(image_path)
                    base_name = image_path.stem.replace("_border_cropped", "")
                    
                    # Use the Stage2Processor for all processing
                    results = self.stage2_processor.process_cropped_table(
                        table_image,
                        base_name,
                        memory_mode=self.stage2_config.memory_mode,
                        debug_dir=self.debug_run_dir_s2
                    )
                    
                    # Collect results
                    refined_tables.extend(results['final_outputs'])
                    output_locations[results['final_stage']].append(base_name)

                except Exception as e:
                    print(f"Error refining {image_path.name}: {e}")
                    if self.stage2_config.save_debug_images:
                        print(f"  [DEBUG] Full error traceback:")
                        traceback.print_exc()
            
            # Store output locations for sequential processing
            self._stage2_output_locations = output_locations

        if self.stage2_config.verbose:
            print(
                f"\n*** STAGE 2 COMPLETE: {len(refined_tables)} publication-ready tables ***"
            )
            
            # Show where outputs ended up (if we have output_locations from sequential processing)
            if hasattr(self, '_stage2_output_locations') and self._stage2_output_locations:
                output_locations = self._stage2_output_locations
                print("\nFinal output locations:")
                if output_locations.get("07_final_deskewed"):
                    print(f"  Final deskewed outputs: {self.stage2_config.output_dir / '07_final_deskewed'} ({len(output_locations['07_final_deskewed'])} images)")
                elif output_locations.get("06_binarized"):
                    print(f"  Binarized outputs: {self.stage2_config.output_dir / '06_binarized'} ({len(output_locations['06_binarized'])} images)")
                elif output_locations.get("05_vertical_strips"):
                    print(f"  Vertical strips: {self.stage2_config.output_dir / '05_vertical_strips'} ({len(output_locations['05_vertical_strips'])} images)")
                elif output_locations.get("04_table_recovered"):
                    print(f"  Table recovered: {self.stage2_config.output_dir / '04_table_recovered'} ({len(output_locations['04_table_recovered'])} images)")
                
                # Show the primary output directory
                primary_stage = max(output_locations.keys(), 
                                  key=lambda k: len(output_locations[k]) if output_locations[k] else 0)
                if output_locations[primary_stage]:
                    print(f"\nPrimary output: {self.stage2_config.output_dir / primary_stage}")
            
            if self.stage2_config.save_debug_images and self.debug_run_dir_s2:
                print(f"\nDebug images saved to: {self.debug_run_dir_s2}")

        # Save run info at completion
        self._save_run_info("stage2", input_dir, len(image_files), "completed")

        return refined_tables

    def _run_stage2_parallel(
        self,
        image_files: List[Path],
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Path]:
        """Run Stage 2 in parallel using multiprocessing.
        
        Args:
            image_files: List of cropped table images to process
            max_workers: Number of worker processes (None = CPU count - 1)
            batch_size: Number of images per batch
            show_progress: Whether to show progress
            
        Returns:
            List of final refined table image paths
        """
        # Set defaults
        if max_workers is None:
            max_workers = self.stage2_config.max_workers or max(1, cpu_count() - 1)
        if batch_size is None:
            # If batch_size is not specified, match it to max_workers for optimal performance
            batch_size = self.stage2_config.batch_size or max_workers
            
        # Create batches
        batches = [
            image_files[i:i + batch_size]
            for i in range(0, len(image_files), batch_size)
        ]
        
        refined_tables = []
        output_locations = {
            "04_table_recovered": [],
            "05_vertical_strips": [],
            "06_binarized": [],
            "07_final_deskewed": []
        }
        successful = 0
        failed = 0
        
        # Create partial function with fixed config
        process_func = partial(
            _process_stage2_worker,
            stage2_config=self.stage2_config,
            debug_dir=self.debug_run_dir_s2
        )
        
        # Try to use progress bar
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
                if hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty():
                    pbar = None
                    print(f"Processing {len(image_files)} tables in parallel (Stage 2)...")
                else:
                    pbar = tqdm(total=len(image_files), desc="Stage 2 Processing", unit="table")
            except:
                pbar = None
                print(f"Processing {len(image_files)} tables in parallel (Stage 2)...")
        
        # Process batches
        for batch in batches:
            with Pool(processes=min(max_workers, len(batch))) as pool:
                results = pool.map(process_func, batch)
                
                for image_path, result_dict, error_msg, elapsed in results:
                    if error_msg:
                        failed += 1
                        if not pbar:
                            print(f"  FAILED {image_path.name}: {error_msg.split(':')[1].split(chr(10))[0]}")
                    else:
                        successful += 1
                        refined_tables.extend(result_dict.get('final_outputs', []))
                        final_stage = result_dict.get('final_stage', '04_table_recovered')
                        base_name = image_path.stem.replace('_border_cropped', '')
                        output_locations[final_stage].append(base_name)
                        
                        if not pbar and self.stage2_config.verbose:
                            print(f"  OK {image_path.name} ({elapsed:.1f}s)")
                    
                    if pbar:
                        pbar.update(1)
        
        if pbar:
            pbar.close()
        
        # Store output locations
        self._stage2_output_locations = output_locations
        
        # Print summary
        if self.stage2_config.verbose:
            print("\n" + "-"*50)
            print("Stage 2 Parallel Processing Summary:")
            print("-"*50)
            print(f"  * Successful: {successful}")
            if failed > 0:
                print(f"  * Failed: {failed}")
            print(f"  * Total refined tables: {len(refined_tables)}")
            print("-"*50)
        
        return refined_tables

    def run(
        self,
        input_path: Path = None,
        use_parallel: Optional[bool] = None,
        use_memory_mode: Optional[bool] = None,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        save_intermediate: bool = False
    ) -> List[Path]:
        """
        Unified method to run the pipeline with flexible options.
        
        Args:
            input_path: Input directory or single image file
            use_parallel: Whether to use parallel processing
            use_memory_mode: Whether to use memory-efficient mode (no intermediate disk I/O)
            max_workers: Number of worker processes (default: CPU count - 1)
            batch_size: Number of images to process per batch
            show_progress: Whether to show progress
            save_intermediate: Whether to save Stage 1 outputs to disk (for debugging)
            
        Returns:
            List of final output paths
        """
        # Track timing
        pipeline_start_time = time.time()
        
        input_path = input_path or self.stage1_config.input_dir
        
        # Use config defaults if not specified
        if use_parallel is None:
            use_parallel = self.stage1_config.parallel_processing or self.stage2_config.parallel_processing
        
        if use_memory_mode is None:
            use_memory_mode = self.stage1_config.memory_mode and self.stage2_config.memory_mode
        
        if batch_size is None:
            batch_size = self.stage1_config.batch_size
            # If config batch_size is also null, match it to max_workers
            if batch_size is None:
                if max_workers is None:
                    max_workers = self.stage1_config.max_workers or max(1, cpu_count() - 1)
                batch_size = max_workers
        
        # Get image files
        if input_path.is_file():
            image_files = [input_path]
        else:
            image_files = get_image_files(input_path)
        
        if not image_files:
            print(f"No images found in {input_path}")
            return []
        
        # Special case: if not using memory mode, run through disk I/O stages
        if not use_memory_mode or save_intermediate:
            print("*** RUNNING COMPLETE TWO-STAGE PIPELINE ***")
            print("=" * 60)
            
            # Check if parallel processing is enabled for disk I/O mode
            if use_parallel:
                print(f"  Mode: Parallel (with disk I/O)")
                print(f"  Workers: {max_workers or max(1, cpu_count() - 1)}")
                print(f"  Batch size: {batch_size}")
                print("  Note: Intermediate files will be saved to disk")
                print("=" * 60)
                
                # Run Stage 1 in parallel with disk saves
                stage1_outputs = self._run_stage1_parallel_disk(
                    input_path,
                    max_workers,
                    batch_size,
                    show_progress
                )
            else:
                print(f"  Mode: Sequential (with disk I/O)")
                # Run Stage 1 sequentially as before
                stage1_outputs = self.run_stage1(input_path)
            
            if not stage1_outputs:
                raise RuntimeError("Stage 1 produced no output. Cannot proceed to Stage 2.")
            
            print("\nStage 1 -> Stage 2 transition")
            print(f"   {len(stage1_outputs)} cropped tables ready for refinement")
            
            # Clean up memory between stages
            gc.collect()
            
            # Run Stage 2 on Stage 1 outputs
            stage1_output_dir = self.stage1_config.output_dir / "07_border_cropped"
            stage2_outputs = self.run_stage2(input_dir=stage1_output_dir)
            
            # Calculate total processing time
            total_time = time.time() - pipeline_start_time
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            print("\nSummary:")
            print(f"  * Total images processed: {len(image_files)}")
            print(f"  * Tables extracted (Stage 1): {len(stage1_outputs)}")
            print(f"  * Tables refined (Stage 2): {len(stage2_outputs)}")
            print(f"  * Processing time: {total_time:.1f} seconds")
            print(f"  * Average per image: {total_time/len(image_files):.1f} seconds")
            
            # Show where final outputs are
            print("\nOutput Locations:")
            if hasattr(self, '_stage2_output_locations'):
                output_locs = self._stage2_output_locations
                active_dirs = []
                if output_locs.get("06_binarized"):
                    active_dirs.append(("Binarized images", f"{self.stage2_config.output_dir / '06_binarized'}"))
                if output_locs.get("05_vertical_strips"):
                    active_dirs.append(("Column strips", f"{self.stage2_config.output_dir / '05_vertical_strips'}"))
                if output_locs.get("04_table_recovered"):
                    active_dirs.append(("Recovered tables", f"{self.stage2_config.output_dir / '04_table_recovered'}"))
                
                if active_dirs:
                    for desc, dir_path in active_dirs:
                        print(f"  * {desc}: {dir_path}")
            else:
                print(f"  * Final results: {self.stage2_config.output_dir}")
            
            print("\n[READY] Tables are prepared for OCR processing!")
            print("="*60)
            
            return stage2_outputs
        
        # Memory mode processing
        print(f"\n*** {'PARALLEL' if use_parallel else 'BATCH'} PROCESSING ***")
        print(f"  Mode: {'Parallel' if use_parallel else 'Sequential'}")
        print(f"  Memory mode: {use_memory_mode}")
        print(f"  Images: {len(image_files)}")
        
        if use_parallel:
            return self._run_batch_parallel(
                image_files, 
                use_memory_mode, 
                max_workers, 
                batch_size,
                show_progress
            )
        else:
            return self._run_batch_sequential(
                image_files, 
                use_memory_mode,
                show_progress
            )
    
    def run_complete(self, input_path: Path = None) -> List[Path]:
        """Backward compatibility: Run both stages sequentially with disk I/O.
        
        Args:
            input_path: Input directory or single image file
            
        Returns:
            List of final refined table image paths
        """
        return self.run(
            input_path=input_path,
            use_memory_mode=False,
            save_intermediate=True
        )
    
    def process_image_memory_mode(self, image_path: Path) -> List[Path]:
        """Backward compatibility: Process a single image in memory mode.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of paths to final output files
        """
        return self.run(
            input_path=image_path,
            use_memory_mode=True,
            use_parallel=False
        )
    
    def run_batch(
        self,
        input_path: Path = None,
        use_parallel: bool = False,
        use_memory_mode: bool = True,
        max_workers: Optional[int] = None,
        batch_size: int = 4,
        show_progress: bool = True
    ) -> List[Path]:
        """Backward compatibility: Redirect to unified run() method.
        
        Args:
            input_path: Input directory or single image file
            use_parallel: Whether to use parallel processing
            use_memory_mode: Whether to use memory-efficient mode
            max_workers: Number of worker processes (default: CPU count - 1)
            batch_size: Number of images to process per batch
            show_progress: Whether to show progress
            
        Returns:
            List of final output paths
        """
        return self.run(
            input_path=input_path,
            use_parallel=use_parallel,
            use_memory_mode=use_memory_mode,
            max_workers=max_workers,
            batch_size=batch_size,
            show_progress=show_progress
        )
    
    def _run_batch_sequential(
        self,
        image_files: List[Path],
        use_memory_mode: bool,
        show_progress: bool
    ) -> List[Path]:
        """Run batch processing sequentially."""
        output_paths = []
        
        for i, image_path in enumerate(image_files, 1):
            if show_progress:
                print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
            
            try:
                if use_memory_mode:
                    # Process in memory
                    stage1_results = self.stage1_processor.process_image(
                        image_path,
                        memory_mode=True,
                        debug_dir=self.debug_run_dir_s1
                    )
                    
                    paths = []
                    for result in stage1_results:
                        base_name = result['page_name'].replace('_border_cropped', '')
                        
                        stage2_result = self.stage2_processor.process_cropped_table(
                            result['image'],
                            base_name,
                            memory_mode=True,
                            debug_dir=self.debug_run_dir_s2
                        )
                        paths.extend(stage2_result.get('final_outputs', []))
                else:
                    # Process with disk I/O
                    stage1_results = self.stage1_processor.process_image(
                        image_path,
                        memory_mode=False,
                        debug_dir=self.debug_run_dir_s1
                    )
                    
                    paths = []
                    for result in stage1_results:
                        table_image = load_image(result['cropped_path'])
                        base_name = result['page_name'].replace('_border_cropped', '')
                        
                        stage2_result = self.stage2_processor.process_cropped_table(
                            table_image,
                            base_name,
                            memory_mode=False,
                            debug_dir=self.debug_run_dir_s2
                        )
                        paths.extend(stage2_result.get('final_outputs', []))
                
                output_paths.extend(paths)
                
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")
        
        print(f"\n*** BATCH COMPLETE: {len(output_paths)} tables processed ***")
        return output_paths
    
    def _run_batch_parallel(
        self,
        image_files: List[Path],
        use_memory_mode: bool,
        max_workers: Optional[int],
        batch_size: Optional[int],
        show_progress: bool
    ) -> List[Path]:
        """Run batch processing in parallel using multiprocessing."""
        # Set max workers
        if max_workers is None:
            max_workers = self.stage1_config.max_workers or max(1, cpu_count() - 1)
        
        # Set batch_size to match max_workers if not specified
        if batch_size is None:
            batch_size = self.stage1_config.batch_size or max_workers
        
        print(f"  Workers: {max_workers}")
        print(f"  Batch size: {batch_size}")
        print("=" * 60)
        
        # Create partial function with fixed configs
        process_func = partial(
            _process_single_image_worker,
            stage1_config=self.stage1_config,
            stage2_config=self.stage2_config,
            stage1_memory_mode=use_memory_mode,
            stage2_memory_mode=use_memory_mode
        )
        
        # Process in batches
        batches = [
            image_files[i:i + batch_size]
            for i in range(0, len(image_files), batch_size)
        ]
        
        output_paths = []
        successful = 0
        failed = 0
        
        # Try to use progress bar
        try:
            from tqdm import tqdm
            # Check if we can use tqdm
            if hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty():
                pbar = None
                print(f"Processing {len(image_files)} images in parallel...")
            else:
                pbar = tqdm(total=len(image_files), desc="Processing images", unit="img")
        except:
            pbar = None
            print(f"Processing {len(image_files)} images in parallel...")
        
        # Process batches
        for batch in batches:
            with Pool(processes=min(max_workers, len(batch))) as pool:
                results = pool.map(process_func, batch)
                
                for image_path, paths, error_msg, elapsed in results:
                    if error_msg:
                        failed += 1
                        if not pbar:
                            print(f"  Failed: {image_path.name}")
                    else:
                        successful += 1
                        output_paths.extend(paths)
                    
                    if pbar:
                        pbar.update(1)
                    elif not show_progress and (successful + failed) % 5 == 0:
                        print(f"  Processed {successful + failed}/{len(image_files)} images...")
        
        if pbar:
            pbar.close()
        
        print(f"\n*** PARALLEL BATCH COMPLETE ***")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total outputs: {len(output_paths)}")
        
        return output_paths
    
    def _run_stage1_parallel_disk(
        self,
        input_path: Path,
        max_workers: Optional[int],
        batch_size: Optional[int],
        show_progress: bool
    ) -> List[Path]:
        """Run Stage 1 in parallel while saving to disk.
        
        Args:
            input_path: Input directory or single image file
            max_workers: Number of worker processes
            batch_size: Images per batch (None = match max_workers)
            show_progress: Whether to show progress
            
        Returns:
            List of paths to cropped table images saved to disk
        """
        # Get image files
        if input_path.is_file():
            image_files = [input_path]
        else:
            image_files = get_image_files(input_path)
        
        if not image_files:
            return []
        
        # Set max workers
        if max_workers is None:
            max_workers = self.stage1_config.max_workers or max(1, cpu_count() - 1)
        
        # Set batch_size to match max_workers if not specified
        if batch_size is None:
            batch_size = self.stage1_config.batch_size or max_workers
        
        print(f"\n*** STAGE 1 PARALLEL PROCESSING ***")
        print(f"  Images: {len(image_files)}")
        print(f"  Workers: {max_workers}")
        print(f"  Output: {self.stage1_config.output_dir}")
        
        # Create partial function with fixed config
        process_func = partial(
            _process_stage1_with_disk_save,
            stage1_config=self.stage1_config,
            debug_dir=self.debug_run_dir_s1
        )
        
        # Process in batches
        batches = [
            image_files[i:i + batch_size]
            for i in range(0, len(image_files), batch_size)
        ]
        
        all_cropped_paths = []
        successful = 0
        failed = 0
        
        # Try to use progress bar
        try:
            from tqdm import tqdm
            if hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty():
                pbar = None
                print(f"Processing {len(image_files)} images in parallel...")
            else:
                pbar = tqdm(total=len(image_files), desc="Stage 1 Processing", unit="img")
        except:
            pbar = None
            print(f"Processing {len(image_files)} images in parallel...")
        
        # Process batches
        memory_failed_images = []
        
        for batch in batches:
            with Pool(processes=min(max_workers, len(batch))) as pool:
                results = pool.map(process_func, batch)
                
                for image_path, cropped_paths, error_msg, elapsed, is_memory_error in results:
                    if error_msg:
                        if is_memory_error:
                            memory_failed_images.append(image_path)
                            if not pbar:
                                print(f"  Memory error: {image_path.name} (will retry sequentially)")
                        else:
                            failed += 1
                            if not pbar:
                                print(f"  Failed: {image_path.name}")
                            if self.stage1_config.verbose:
                                print(f"    Error: {error_msg}")
                    else:
                        successful += 1
                        if cropped_paths:
                            all_cropped_paths.extend(cropped_paths)
                    
                    if pbar:
                        pbar.update(1)
                    elif not show_progress and (successful + failed + len(memory_failed_images)) % 5 == 0:
                        print(f"  Processed {successful + failed + len(memory_failed_images)}/{len(image_files)} images...")
        
        if pbar:
            pbar.close()
        
        # Retry memory-failed images sequentially
        if memory_failed_images:
            print(f"\n*** RETRYING {len(memory_failed_images)} IMAGES SEQUENTIALLY ***")
            print("  Processing large images one at a time to avoid memory errors...")
            
            for image_path in memory_failed_images:
                print(f"  Retrying: {image_path.name}")
                try:
                    # Force garbage collection before processing large image
                    gc.collect()
                    
                    # Process single image
                    result = process_func(image_path)
                    image_path, cropped_paths, error_msg, elapsed, is_memory_error = result
                    
                    if error_msg:
                        failed += 1
                        print(f"    Failed again: {error_msg}")
                    else:
                        successful += 1
                        if cropped_paths:
                            all_cropped_paths.extend(cropped_paths)
                        print(f"    Success! Processed in {elapsed:.1f}s")
                    
                    # Clean up memory after each large image
                    gc.collect()
                    
                except Exception as e:
                    failed += 1
                    print(f"    Failed with exception: {str(e)}")
        
        print(f"\n*** STAGE 1 PARALLEL COMPLETE ***")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Cropped tables: {len(all_cropped_paths)}")
        
        return all_cropped_paths
    
    # Compatibility methods for backward compatibility
    def run_complete_pipeline(self, input_path: Path = None) -> List[Path]:
        """Backward compatibility method."""
        return self.run_complete(input_path)
    
    def run_batch_optimized(
        self,
        input_path: Path = None,
        use_parallel: bool = None,
        use_memory_mode: bool = None,
        stage1_memory_mode: bool = None,
        stage2_memory_mode: bool = None
    ) -> List[Path]:
        """Backward compatibility method for run_batch_optimized."""
        # Handle memory mode parameters
        if stage1_memory_mode is not None or stage2_memory_mode is not None:
            memory_mode = stage1_memory_mode or stage2_memory_mode or use_memory_mode
        else:
            memory_mode = use_memory_mode
        
        # Use config values if not specified
        if use_parallel is None:
            use_parallel = self.stage1_config.parallel_processing or self.stage2_config.parallel_processing
        
        if memory_mode is None:
            memory_mode = self.stage1_config.memory_mode or self.stage2_config.memory_mode
        
        return self.run_batch(
            input_path=input_path,
            use_parallel=use_parallel,
            use_memory_mode=memory_mode
        )


def main() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="OCR Table Extraction Pipeline")
    parser.add_argument(
        "input", nargs="?", help="Input directory or file (default: use config)"
    )
    parser.add_argument("-o", "--output", help="Output directory (default: use config)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug images")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug images")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--memory", action="store_true", help="Enable memory mode")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory mode (use disk I/O)")
    parser.add_argument("--save-intermediate", action="store_true", help="Save Stage 1 outputs to disk")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    
    # Stage selection options
    parser.add_argument("--stage1-only", action="store_true", help="Run only Stage 1 (initial processing)")
    parser.add_argument("--stage2-only", action="store_true", help="Run only Stage 2 (refinement)")

    args = parser.parse_args()
    
    # Validate stage selection
    if args.stage1_only and args.stage2_only:
        print("Error: Cannot specify both --stage1-only and --stage2-only")
        sys.exit(1)

    # Load configurations with all defaults from config files
    stage1_config = get_stage1_config()
    stage2_config = get_stage2_config()
    
    # Only override paths if user provided them
    if args.input:
        stage1_config.input_dir = Path(args.input)
        input_path = Path(args.input)
    else:
        input_path = stage1_config.input_dir
    
    if args.output:
        stage1_config.output_dir = Path(args.output) / "stage1"
        stage2_config.output_dir = Path(args.output) / "stage2"
    # else: keep config defaults (data/output/stage1 and data/output/stage2)
    
    # Only override verbose if user explicitly set it
    if args.verbose:
        stage1_config.verbose = True
        stage2_config.verbose = True
    elif args.quiet:
        stage1_config.verbose = False
        stage2_config.verbose = False
    # else: keep config defaults (currently True in both configs)
    
    # Only override debug if user explicitly set it
    if args.debug:
        stage1_config.save_debug_images = True
        stage2_config.save_debug_images = True
    elif args.no_debug:
        stage1_config.save_debug_images = False
        stage2_config.save_debug_images = False
    # else: keep config defaults (currently False in both configs)

    # Create pipeline
    pipeline = TwoStageOCRPipeline(stage1_config, stage2_config)

    # Determine processing options (None means use config default)
    use_parallel = None
    if args.parallel:
        use_parallel = True
    elif args.no_parallel:
        use_parallel = False
    
    use_memory = None
    if args.memory:
        use_memory = True
    elif args.no_memory:
        use_memory = False
    
    # Run based on stage selection
    if args.stage1_only:
        # Run only Stage 1
        outputs = pipeline.run_stage1(input_path)
        print("\n" + "-"*50)
        print("[STAGE 1 COMPLETE]")
        print("-"*50)
        print(f"Results:")
        print(f"  * Cropped tables extracted: {len(outputs)}")
        print(f"  * Output location: {stage1_config.output_dir / '07_border_cropped'}")
        print("\nNext: Run Stage 2 for table refinement or use the cropped tables directly.")
    elif args.stage2_only:
        # Run only Stage 2 (uses Stage 1 output as input)
        stage2_input = stage2_config.input_dir or (stage1_config.output_dir / "07_border_cropped")
        if not stage2_input.exists():
            print(f"Error: Stage 1 output not found: {stage2_input}")
            print("Run Stage 1 first or use complete pipeline")
            sys.exit(1)
        outputs = pipeline.run_stage2(stage2_input)
        print("\n" + "-"*50)
        print("[STAGE 2 COMPLETE]")
        print("-"*50)
        print(f"Results:")
        print(f"  * Refined tables: {len(outputs)}")
        print(f"  * Output location: {stage2_config.output_dir}")
        print("\n[READY] Tables are refined and ready for OCR processing!")
    else:
        # Run complete pipeline with unified method - None values will use config defaults
        outputs = pipeline.run(
            input_path,
            use_parallel=use_parallel,
            use_memory_mode=use_memory,
            max_workers=args.workers,
            batch_size=args.batch_size,
            save_intermediate=args.save_intermediate
        )
        if outputs:
            print("\n" + "="*60)
            print("[PIPELINE COMPLETE]")
            print(f"Successfully processed {len(outputs)} tables")
            print("="*60)
        else:
            print("\n[WARNING] No tables were processed. Check your input directory and configuration.")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    from multiprocessing import freeze_support
    freeze_support()
    
    main()