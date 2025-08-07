"""Memory-efficient pipeline for processing images without intermediate disk I/O."""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import json
from datetime import datetime

from .config import Stage1Config, Stage2Config, get_stage1_config, get_stage2_config
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
    TableDetectionProcessor,
    DeskewProcessor,
    PageSplitProcessor,
    MarginRemovalProcessor,
    MarkRemovalProcessor,
    TagRemovalProcessor,
    TableProcessingProcessor,
    BinarizeProcessor,
)
from .processors.table_recovery_utils import get_major_vertical_boundaries


class MemoryEfficientPipeline:
    """Pipeline that processes images in memory to reduce disk I/O."""
    
    def __init__(
        self,
        stage1_config: Stage1Config = None,
        stage2_config: Stage2Config = None,
        memory_mode: bool = True
    ):
        """Initialize memory-efficient pipeline.
        
        Args:
            stage1_config: Configuration for Stage 1
            stage2_config: Configuration for Stage 2
            memory_mode: If True, keeps intermediate results in memory
        """
        self.stage1_config = stage1_config or get_stage1_config()
        self.stage2_config = stage2_config or get_stage2_config()
        self.memory_mode = memory_mode
        
        # Store intermediate results in memory
        self.memory_cache = {}
        
        # Initialize processors for debug mode
        self._init_processors()
        
        # Generate timestamp for run
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._init_debug_dirs()
    
    def _init_processors(self):
        """Initialize processors for both stages."""
        self.table_detector_s1 = TableDetectionProcessor(self.stage1_config)
        self.table_detector_s2 = TableDetectionProcessor(self.stage2_config)
        self.deskew_processor_s1 = DeskewProcessor(self.stage1_config)
        self.deskew_processor_s2 = DeskewProcessor(self.stage2_config)
        self.margin_processor_s1 = MarginRemovalProcessor(self.stage1_config)
        self.page_split_processor_s1 = PageSplitProcessor(self.stage1_config)
        self.mark_removal_processor_s1 = MarkRemovalProcessor(self.stage1_config)
        self.tag_removal_processor_s1 = TagRemovalProcessor(self.stage1_config)
        self.table_processing_processor_s1 = TableProcessingProcessor(self.stage1_config)
        self.binarize_processor_s2 = BinarizeProcessor(self.stage2_config)
    
    def _init_debug_dirs(self):
        """Initialize debug directories if needed."""
        self.debug_run_dir_s1 = None
        self.debug_run_dir_s2 = None
        self.debug_base_dir = None
        
        # Create a common base directory for this run if debug is enabled for either stage
        if ((self.stage1_config.save_debug_images and self.stage1_config.debug_dir) or 
            (self.stage2_config.save_debug_images and self.stage2_config.debug_dir)):
            # Use the first available debug_dir as the base
            base_debug_dir = self.stage1_config.debug_dir if self.stage1_config.debug_dir else self.stage2_config.debug_dir
            self.debug_base_dir = base_debug_dir / f"{self.run_timestamp}_run"
            self.debug_base_dir.mkdir(parents=True, exist_ok=True)
        
        if self.stage1_config.save_debug_images and self.stage1_config.debug_dir:
            self.debug_run_dir_s1 = self.debug_base_dir / "stage1"
            self.debug_run_dir_s1.mkdir(parents=True, exist_ok=True)
            
        if self.stage2_config.save_debug_images and self.stage2_config.debug_dir:
            self.debug_run_dir_s2 = self.debug_base_dir / "stage2"
            self.debug_run_dir_s2.mkdir(parents=True, exist_ok=True)
    
    def _save_if_needed(self, image: np.ndarray, path: Path, condition: bool = True) -> None:
        """Save image to disk only if needed (not in memory mode or explicitly requested).
        
        Args:
            image: Image to save
            path: Path to save to
            condition: Additional condition for saving
        """
        if not self.memory_mode or condition:
            save_image(image, path)
    
    def _cache_result(self, key: str, data: Any) -> None:
        """Cache intermediate result in memory.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        if self.memory_mode:
            self.memory_cache[key] = data
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result from memory.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None
        """
        return self.memory_cache.get(key)
    
    def process_single_image_stage1(
        self,
        image_path: Path
    ) -> Dict[str, Any]:
        """Process a single image through Stage 1 in memory.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing processed results for each page
        """
        # Show file name in verbose or debug mode
        if self.stage1_config.verbose or self.stage1_config.save_debug_images:
            print(f"\nProcessing: {image_path.name}")
        
        # Load image once
        image = load_image(image_path)
        processing_image = image
        
        # Step 1: Mark removal (on full image)
        if self.stage1_config.enable_mark_removal:
            table_lines_mask = None
            
            if self.stage1_config.mark_removal_protect_table_lines:
                h_lines, v_lines = detect_table_lines(
                    processing_image,
                    threshold=self.stage1_config.mark_removal_table_threshold,
                    horizontal_kernel_size=self.stage1_config.mark_removal_table_h_kernel,
                    vertical_kernel_size=self.stage1_config.mark_removal_table_v_kernel,
                )
                
                if h_lines or v_lines:
                    table_lines_mask = create_table_lines_mask(
                        processing_image.shape,
                        h_lines,
                        v_lines,
                        line_thickness=self.stage1_config.mark_removal_table_line_thickness
                    )
            
            processing_image = remove_marks(
                processing_image,
                dilate_iter=self.stage1_config.mark_removal_dilate_iter,
                kernel_size=self.stage1_config.mark_removal_kernel_size,
                table_lines_mask=table_lines_mask
            )
            
            # Only save if intermediate outputs are requested
            if self.stage1_config.save_intermediate_outputs:
                marks_dir = self.stage1_config.output_dir / "01_marks_removed"
                marks_path = marks_dir / f"{image_path.stem}_marks_removed.jpg"
                self._save_if_needed(processing_image, marks_path)
        
        # Step 2: Margin removal
        if self.stage1_config.enable_margin_removal:
            margin_params = {
                'blur_ksize': getattr(self.stage1_config, 'margin_blur_ksize', 20),
                'close_ksize': getattr(self.stage1_config, 'margin_close_ksize', 30),
                'close_iter': getattr(self.stage1_config, 'margin_close_iter', 5),
                'erode_after_close': getattr(self.stage1_config, 'margin_erode_after_close', 0),
                'use_gradient_detection': getattr(self.stage1_config, 'margin_use_gradient_detection', False),
                'gradient_threshold': getattr(self.stage1_config, 'margin_gradient_threshold', 30),
            }
            
            processing_image = remove_margin_inscribed(processing_image, **margin_params)
            
            if self.stage1_config.save_intermediate_outputs:
                margin_dir = self.stage1_config.output_dir / "02_margin_removed"
                margin_path = margin_dir / f"{image_path.stem}_margin_removed.jpg"
                self._save_if_needed(processing_image, margin_path)
        
        # Step 3: Split into pages
        if self.stage1_config.enable_page_splitting:
            right_page, left_page = split_two_page_image(
                processing_image,
                search_ratio=self.stage1_config.search_ratio,
                line_len_frac=getattr(self.stage1_config, 'line_len_frac', 0.3),
                line_thick=getattr(self.stage1_config, 'line_thick', 3),
                peak_thr=getattr(self.stage1_config, 'peak_thr', 0.3),
            )
            pages_to_process = [("left", left_page), ("right", right_page)]
        else:
            pages_to_process = [("full", processing_image)]
        
        # Process each page and store results
        page_results = {}
        
        for page_suffix, page in pages_to_process:
            page_name = f"{image_path.stem}_{page_suffix}"
            page_image = page
            
            # Step 4: Tag removal (per page)
            if self.stage1_config.enable_tag_removal:
                page_image = remove_tags(
                    page_image,
                    thresh_dark=self.stage1_config.tag_removal_thresh_dark,
                    row_sum_thresh=self.stage1_config.tag_removal_row_sum_thresh,
                    dark_ratio=self.stage1_config.tag_removal_dark_ratio,
                    min_area=self.stage1_config.tag_removal_min_area,
                    max_area=self.stage1_config.tag_removal_max_area,
                    min_aspect=self.stage1_config.tag_removal_min_aspect,
                    max_aspect=self.stage1_config.tag_removal_max_aspect,
                )
            
            # Step 5: Deskew
            if self.stage1_config.enable_deskewing:
                deskewed, angle = deskew_image(
                    page_image,
                    coarse_range=self.stage1_config.coarse_range,
                    coarse_step=self.stage1_config.coarse_step,
                    fine_range=self.stage1_config.fine_range,
                    fine_step=self.stage1_config.fine_step,
                    min_angle_correction=self.stage1_config.min_angle_correction,
                )
                
                # Deskewing completed
            else:
                deskewed = page_image
            
            # Step 6: Table line detection
            h_lines, v_lines = detect_table_lines(
                deskewed,
                threshold=self.stage1_config.threshold,
                horizontal_kernel_size=self.stage1_config.horizontal_kernel_size,
                vertical_kernel_size=self.stage1_config.vertical_kernel_size,
                alignment_threshold=self.stage1_config.alignment_threshold,
                h_min_length_image_ratio=self.stage1_config.h_min_length_image_ratio,
                h_min_length_relative_ratio=self.stage1_config.h_min_length_relative_ratio,
                v_min_length_image_ratio=self.stage1_config.v_min_length_image_ratio,
                v_min_length_relative_ratio=self.stage1_config.v_min_length_relative_ratio,
                min_aspect_ratio=self.stage1_config.min_aspect_ratio,
                max_h_length_ratio=self.stage1_config.max_h_length_ratio,
                max_v_length_ratio=self.stage1_config.max_v_length_ratio,
                close_line_distance=self.stage1_config.close_line_distance,
                search_region_top=self.stage1_config.search_region_top,
                search_region_bottom=self.stage1_config.search_region_bottom,
                search_region_left=self.stage1_config.search_region_left,
                search_region_right=self.stage1_config.search_region_right,
            )
            
            # Step 7: Table structure detection
            table_structure, _ = detect_table_structure(
                h_lines,
                v_lines,
                eps=getattr(self.stage1_config, 'table_detection_eps', 10),
                return_analysis=True
            )
            
            # Step 8: Crop to table borders
            if self.stage1_config.enable_table_cropping and table_structure.get("cells"):
                cropped_table = crop_to_table_borders(
                    deskewed,
                    table_structure,
                    padding=getattr(self.stage1_config, 'table_crop_padding', 20),
                    return_analysis=False
                )
            else:
                cropped_table = deskewed
            
            # Store results in memory
            page_results[page_suffix] = {
                'image': cropped_table,
                'h_lines': h_lines,
                'v_lines': v_lines,
                'table_structure': table_structure,
                'page_name': page_name
            }
            
            # Save final cropped table (always save this as it's the Stage 1 output)
            crop_dir = self.stage1_config.output_dir / "07_border_cropped"
            crop_path = crop_dir / f"{page_name}_border_cropped.jpg"
            save_image(cropped_table, crop_path)
            
            # Save metadata (JSON files are small, always save)
            self._save_metadata(page_name, h_lines, v_lines, table_structure)
            
            # Page processed successfully
        
        return page_results
    
    def _save_metadata(
        self,
        page_name: str,
        h_lines: List,
        v_lines: List,
        table_structure: Dict
    ) -> None:
        """Save metadata JSON files (lines and structure data).
        
        Args:
            page_name: Name of the page
            h_lines: Horizontal lines
            v_lines: Vertical lines
            table_structure: Table structure data
        """
        # Save lines data
        lines_dir = self.stage1_config.output_dir / "05_table_lines"
        lines_data_dir = lines_dir / "lines_data"
        lines_data_dir.mkdir(parents=True, exist_ok=True)
        lines_json_path = lines_data_dir / f"{page_name}_lines_data.json"
        lines_data = {
            "horizontal_lines": [[int(x) for x in line] for line in h_lines],
            "vertical_lines": [[int(x) for x in line] for line in v_lines]
        }
        with open(lines_json_path, 'w') as f:
            json.dump(lines_data, f, indent=2)
        
        # Save structure data
        structure_dir = self.stage1_config.output_dir / "06_table_structure"
        structure_data_dir = structure_dir / "structure_data"
        structure_data_dir.mkdir(parents=True, exist_ok=True)
        structure_json_path = structure_data_dir / f"{page_name}_structure_data.json"
        with open(structure_json_path, 'w') as f:
            json.dump(table_structure, f, indent=2)
    
    def process_single_image_stage2(
        self,
        cropped_table: np.ndarray,
        base_name: str,
        lines_data: Optional[Dict] = None,
        structure_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process a single cropped table through Stage 2 in memory.
        
        Args:
            cropped_table: Cropped table image from Stage 1
            base_name: Base name for output files
            lines_data: Optional pre-computed lines data
            structure_data: Optional pre-computed structure data
            
        Returns:
            Dictionary containing processed results
        """
        # Show stage 2 processing in verbose or debug mode
        if self.stage2_config.verbose or self.stage2_config.save_debug_images:
            print(f"  Refining: {base_name}")
        
        # Process Stage 2 refinement
        
        # Step 1: Re-deskew for fine-tuning
        if self.stage2_config.enable_deskewing:
            refined_deskewed, angle = deskew_image(
                cropped_table,
                coarse_range=self.stage2_config.coarse_range,
                coarse_step=self.stage2_config.coarse_step,
                fine_range=self.stage2_config.fine_range,
                fine_step=self.stage2_config.fine_step,
                min_angle_correction=self.stage2_config.min_angle_correction,
            )
        else:
            refined_deskewed = cropped_table
        
        # Step 2: Refined line detection (use provided or detect new)
        if lines_data:
            h_lines = lines_data.get('horizontal_lines', [])
            v_lines = lines_data.get('vertical_lines', [])
        else:
            h_lines, v_lines = detect_table_lines(
                refined_deskewed,
                threshold=self.stage2_config.threshold,
                horizontal_kernel_size=self.stage2_config.horizontal_kernel_size,
                vertical_kernel_size=self.stage2_config.vertical_kernel_size,
                alignment_threshold=self.stage2_config.alignment_threshold,
                h_min_length_image_ratio=self.stage2_config.h_min_length_image_ratio,
                h_min_length_relative_ratio=self.stage2_config.h_min_length_relative_ratio,
                v_min_length_image_ratio=self.stage2_config.v_min_length_image_ratio,
                v_min_length_relative_ratio=self.stage2_config.v_min_length_relative_ratio,
                min_aspect_ratio=self.stage2_config.min_aspect_ratio,
                max_h_length_ratio=self.stage2_config.max_h_length_ratio,
                max_v_length_ratio=self.stage2_config.max_v_length_ratio,
                close_line_distance=self.stage2_config.close_line_distance,
            )
        
        # Step 3: Table structure detection (use provided or detect new)
        if structure_data:
            table_structure = structure_data
        else:
            table_structure, _ = detect_table_structure(
                h_lines,
                v_lines,
                eps=getattr(self.stage2_config, 'table_detection_eps', 10),
                return_analysis=True
            )
        
        # Save metadata for table recovery (needed for the function call)
        lines_dir = self.stage2_config.output_dir / "02_table_lines"
        lines_data_dir = lines_dir / "lines_data"
        lines_data_dir.mkdir(parents=True, exist_ok=True)
        lines_json_path = lines_data_dir / f"{base_name}_lines_data.json"
        with open(lines_json_path, 'w') as f:
            json.dump({
                "horizontal_lines": [[int(x) for x in line] for line in h_lines],
                "vertical_lines": [[int(x) for x in line] for line in v_lines]
            }, f, indent=2)
        
        structure_dir = self.stage2_config.output_dir / "03_table_structure"
        structure_data_dir = structure_dir / "structure_data"
        structure_data_dir.mkdir(parents=True, exist_ok=True)
        structure_json_path = structure_data_dir / f"{base_name}_structure_data.json"
        with open(structure_json_path, 'w') as f:
            json.dump(table_structure, f, indent=2)
        
        # Step 4: Table recovery
        if table_structure.get("cells"):
            recovered_result = table_recovery(
                lines_json_path=str(lines_json_path),
                structure_json_path=str(structure_json_path),
                coverage_ratio=getattr(self.stage2_config, 'table_recovery_coverage_ratio', 0.8),
                background_image=refined_deskewed,
                highlight_merged=True,
                show_grid=True,
                label_cells=True
            )
        else:
            recovered_result = {
                'visualization': refined_deskewed,
                'rows': [],
                'cols': [],
                'cells': []
            }
        
        # Save final recovered table (always save as this is the final output)
        recovery_dir = self.stage2_config.output_dir / "04_table_recovered"
        recovery_path = recovery_dir / f"{base_name}_table_recovered.jpg"
        save_image(recovered_result['visualization'], recovery_path)
        
        # Save recovered structure JSON
        recovery_json_dir = recovery_dir / "recovery_data"
        recovery_json_dir.mkdir(parents=True, exist_ok=True)
        recovery_json_path = recovery_json_dir / f"{base_name}_recovered.json"
        with open(recovery_json_path, 'w') as f:
            json.dump({
                'rows': recovered_result['rows'],
                'cols': recovered_result['cols'],
                'cells': recovered_result['cells']
            }, f, indent=2)
        
        # Table recovery completed
        
        return {
            'refined_image': refined_deskewed,
            'recovered_result': recovered_result,
            'recovery_path': recovery_path
        }
    
    def process_image_complete(self, image_path: Path) -> List[Path]:
        """Process a single image through both stages in memory.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of paths to final output files
        """
        # Stage 1: Process and get results in memory
        stage1_results = self.process_single_image_stage1(image_path)
        
        output_paths = []
        
        # Stage 2: Process each page result from memory
        for page_suffix, page_data in stage1_results.items():
            base_name = page_data['page_name'].replace('_border_cropped', '')
            
            # Process Stage 2 using in-memory data
            stage2_result = self.process_single_image_stage2(
                page_data['image'],
                base_name,
                lines_data={
                    'horizontal_lines': page_data['h_lines'],
                    'vertical_lines': page_data['v_lines']
                },
                structure_data=page_data['table_structure']
            )
            
            output_paths.append(stage2_result['recovery_path'])
        
        return output_paths
    
    def clear_cache(self) -> None:
        """Clear the memory cache to free up memory."""
        self.memory_cache.clear()