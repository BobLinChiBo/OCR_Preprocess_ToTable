"""Parallel processing pipeline for batch image processing."""

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback
from tqdm import tqdm
import json

from .memory_pipeline import MemoryEfficientPipeline
from .config import Stage1Config, Stage2Config, get_stage1_config, get_stage2_config
from .processors import get_image_files


def process_single_image_wrapper(
    image_path: Path,
    stage1_config: Stage1Config,
    stage2_config: Stage2Config,
    memory_mode: bool = True
) -> Tuple[Path, Optional[List[Path]], Optional[str]]:
    """Wrapper function for processing a single image in a worker process.
    
    Args:
        image_path: Path to the image
        stage1_config: Stage 1 configuration
        stage2_config: Stage 2 configuration
        memory_mode: Whether to use memory mode
        
    Returns:
        Tuple of (image_path, output_paths, error_message)
    """
    try:
        # Create a new pipeline instance in the worker process
        pipeline = MemoryEfficientPipeline(
            stage1_config=stage1_config,
            stage2_config=stage2_config,
            memory_mode=memory_mode
        )
        
        # Process the image
        output_paths = pipeline.process_image_complete(image_path)
        
        # Clear cache to free memory before returning
        pipeline.clear_cache()
        
        return (image_path, output_paths, None)
    except Exception as e:
        error_msg = f"Error processing {image_path}: {str(e)}\n{traceback.format_exc()}"
        return (image_path, None, error_msg)


class ParallelPipeline:
    """Pipeline for parallel batch processing of images."""
    
    def __init__(
        self,
        stage1_config: Stage1Config = None,
        stage2_config: Stage2Config = None,
        max_workers: Optional[int] = None,
        batch_size: int = 4,
        memory_mode: bool = True,
        show_progress: bool = True
    ):
        """Initialize parallel pipeline.
        
        Args:
            stage1_config: Configuration for Stage 1
            stage2_config: Configuration for Stage 2
            max_workers: Maximum number of worker processes (default: CPU count - 1)
            batch_size: Number of images to process in each batch
            memory_mode: Whether to use memory-efficient mode
            show_progress: Whether to show progress bar
        """
        self.stage1_config = stage1_config or get_stage1_config()
        self.stage2_config = stage2_config or get_stage2_config()
        
        # Set max workers, leaving one CPU for the main process
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self.batch_size = batch_size
        self.memory_mode = memory_mode
        self.show_progress = show_progress
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Track results
        self.successful_results = []
        self.failed_results = []
    
    def _ensure_directories(self):
        """Ensure all output directories exist."""
        # Stage 1 directories
        if self.stage1_config.output_dir:
            for subdir in [
                "01_marks_removed",
                "02_margin_removed", 
                "03_page_split",
                "04_deskewed",
                "05_table_lines",
                "05_table_lines/lines_data",
                "06_table_structure",
                "06_table_structure/structure_data",
                "07_border_cropped"
            ]:
                (self.stage1_config.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Stage 2 directories
        if self.stage2_config.output_dir:
            for subdir in [
                "01_refined_deskewed",
                "02_table_lines",
                "02_table_lines/lines_data",
                "03_table_structure",
                "03_table_structure/structure_data",
                "04_table_recovered",
                "04_table_recovered/recovery_data"
            ]:
                (self.stage2_config.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def process_batch(
        self,
        input_images: List[Path],
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Process a batch of images.
        
        Args:
            input_images: List of image paths to process
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary with processing results
        """
        if not input_images:
            return {
                'successful': [],
                'failed': [],
                'total': 0
            }
        
        # Reset results
        self.successful_results = []
        self.failed_results = []
        
        if parallel and len(input_images) > 1:
            return self._process_parallel(input_images)
        else:
            return self._process_sequential(input_images)
    
    def _process_parallel(self, input_images: List[Path]) -> Dict[str, Any]:
        """Process images in parallel using multiprocessing.
        
        Args:
            input_images: List of image paths
            
        Returns:
            Dictionary with processing results
        """
        print(f"\nProcessing {len(input_images)} images using {self.max_workers} workers...")
        
        # Create partial function with fixed configs
        process_func = partial(
            process_single_image_wrapper,
            stage1_config=self.stage1_config,
            stage2_config=self.stage2_config,
            memory_mode=self.memory_mode
        )
        
        # Process in batches to avoid memory issues
        batches = [
            input_images[i:i + self.batch_size]
            for i in range(0, len(input_images), self.batch_size)
        ]
        
        # Always show progress bar for better user experience
        pbar = tqdm(total=len(input_images), desc="Processing images", unit="img")
        
        # Process each batch
        for batch in batches:
            with Pool(processes=min(self.max_workers, len(batch))) as pool:
                results = pool.map(process_func, batch)
                
                # Process results
                for image_path, output_paths, error_msg in results:
                    if error_msg:
                        self.failed_results.append({
                            'image': str(image_path),
                            'error': error_msg
                        })
                        # Don't print individual failures when using progress bar
                        pass
                    else:
                        self.successful_results.append({
                            'image': str(image_path),
                            'outputs': [str(p) for p in output_paths]
                        })
                    
                    pbar.update(1)
        
        pbar.close()
        
        # Generate summary
        return self._generate_summary()
    
    def _process_sequential(self, input_images: List[Path]) -> Dict[str, Any]:
        """Process images sequentially (fallback for single image or debugging).
        
        Args:
            input_images: List of image paths
            
        Returns:
            Dictionary with processing results
        """
        print(f"\nProcessing {len(input_images)} images sequentially...")
        
        # Create pipeline instance
        pipeline = MemoryEfficientPipeline(
            stage1_config=self.stage1_config,
            stage2_config=self.stage2_config,
            memory_mode=self.memory_mode
        )
        
        # Always use progress bar for better experience
        pbar = tqdm(input_images, desc="Processing images", unit="img")
        images_to_process = pbar
        
        # Process each image
        for image_path in images_to_process:
            try:
                output_paths = pipeline.process_image_complete(image_path)
                self.successful_results.append({
                    'image': str(image_path),
                    'outputs': [str(p) for p in output_paths]
                })
                    
                # Clear cache after each image to free memory
                pipeline.clear_cache()
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.failed_results.append({
                    'image': str(image_path),
                    'error': error_msg
                })
        
        # Generate summary
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate processing summary.
        
        Returns:
            Dictionary with summary statistics
        """
        total = len(self.successful_results) + len(self.failed_results)
        success_rate = (len(self.successful_results) / total * 100) if total > 0 else 0
        
        summary = {
            'successful': self.successful_results,
            'failed': self.failed_results,
            'total': total,
            'success_count': len(self.successful_results),
            'failed_count': len(self.failed_results),
            'success_rate': f"{success_rate:.1f}%"
        }
        
        # Print summary
        print("\n" + "="*50)
        print("Processing Summary:")
        print(f"  Total images: {total}")
        print(f"  Successful: {len(self.successful_results)}")
        print(f"  Failed: {len(self.failed_results)}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        if self.failed_results and self.stage1_config.verbose:
            print("\nFailed images:")
            for failed in self.failed_results:
                print(f"  - {Path(failed['image']).name}")
        
        print("="*50)
        
        return summary
    
    def save_summary(self, summary: Dict[str, Any], output_path: Optional[Path] = None):
        """Save processing summary to JSON file.
        
        Args:
            summary: Summary dictionary
            output_path: Path to save summary (default: output_dir/processing_summary.json)
        """
        if output_path is None:
            output_path = self.stage2_config.output_dir / "processing_summary.json"
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {output_path}")
    
    def process_directory(
        self,
        input_dir: Path,
        pattern: str = "*.jpg",
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            pattern: File pattern to match (default: "*.jpg")
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary with processing results
        """
        # Get all image files
        image_files = get_image_files(input_dir, pattern)
        
        if not image_files:
            print(f"No images found in {input_dir} with pattern {pattern}")
            return {
                'successful': [],
                'failed': [],
                'total': 0
            }
        
        print(f"Found {len(image_files)} images to process")
        
        # Process the batch
        summary = self.process_batch(image_files, parallel=parallel)
        
        # Save summary
        self.save_summary(summary)
        
        return summary