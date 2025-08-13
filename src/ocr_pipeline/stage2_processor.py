"""Unified Stage 2 processor for all pipeline implementations."""

import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import json

from .config import Stage2Config
from .processors import (
    load_image,
    save_image,
    detect_table_lines,
    visualize_detected_lines,
    detect_table_structure,
    visualize_table_structure,
    table_recovery,
    cut_vertical_strips,
    TableDetectionProcessor,
    DeskewProcessor,
    BinarizeProcessor,
)
from .processors.table_recovery_utils import get_all_vertical_boundaries


class Stage2Processor:
    """Unified processor for Stage 2 of the OCR pipeline.
    
    This class contains the complete Stage 2 processing logic that is shared
    across all pipeline implementations (regular, memory-efficient, and parallel).
    """
    
    def __init__(self, config: Stage2Config):
        """Initialize Stage 2 processor with configuration.
        
        Args:
            config: Stage 2 configuration
        """
        self.config = config
        
        # Initialize processors
        self.table_detector = TableDetectionProcessor(config)
        self.deskew_processor = DeskewProcessor(config)
        self.binarize_processor = BinarizeProcessor(config)
    
    def process_cropped_table(
        self,
        cropped_table: np.ndarray,
        base_name: str,
        memory_mode: bool = False,
        debug_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Process a single cropped table through Stage 2.
        
        Args:
            cropped_table: Cropped table image from Stage 1
            base_name: Base name for output files
            memory_mode: If True, keeps results in memory instead of saving to disk
            debug_dir: Optional directory for debug images
            
        Returns:
            Dictionary containing processed results
        """
        # Show image name when verbose or debug mode is enabled
        if self.config.verbose or self.config.save_debug_images:
            print(f"\nRefining: {base_name}")
        
        image_start_time = time.time()
        
        # Step 1: Re-deskew for fine-tuning
        refined_deskewed = self._process_deskew(
            cropped_table, base_name, memory_mode, debug_dir
        )
        
        # Step 2: Refined table line detection (with optional internal preprocessing)
        h_lines, v_lines, lines_json_path = self._process_table_detection(
            refined_deskewed, base_name, memory_mode, debug_dir
        )
        
        # Step 3: Table structure detection
        table_structure, structure_json_path = self._process_table_structure(
            refined_deskewed, h_lines, v_lines, base_name, memory_mode
        )
        
        # Step 4: Table recovery
        recovered_result = self._process_table_recovery(
            refined_deskewed, lines_json_path, structure_json_path,
            table_structure, base_name, memory_mode
        )
        
        # Step 5: Vertical strip cutting (optional)
        strips_result = self._process_vertical_strips(
            refined_deskewed, recovered_result, base_name, memory_mode
        )
        
        # Step 6: Final binarization (optional)
        binarized_paths = self._process_binarization(
            strips_result, base_name, memory_mode, debug_dir
        )
        
        # Step 7: Final deskew (optional final step)
        final_deskewed_paths = self._process_final_deskew(
            binarized_paths, strips_result, recovered_result,
            base_name, memory_mode, debug_dir
        )
        
        # Print total timing
        total_elapsed = time.time() - image_start_time
        print(f"    [TIMING] Total processing time: {total_elapsed:.3f} seconds")
        
        # Determine final output stage
        if final_deskewed_paths:
            final_stage = "07_final_deskewed"
            final_outputs = final_deskewed_paths
        elif binarized_paths:
            final_stage = "06_binarized"
            final_outputs = binarized_paths
        elif strips_result and strips_result.get('strip_paths'):
            final_stage = "05_vertical_strips"
            final_outputs = strips_result['strip_paths']
        else:
            final_stage = "04_table_recovered"
            final_outputs = [self.config.output_dir / "04_table_recovered" / f"{base_name}_table_recovered.jpg"]
        
        return {
            'base_name': base_name,
            'refined_deskewed': refined_deskewed,
            'h_lines': h_lines,
            'v_lines': v_lines,
            'table_structure': table_structure,
            'recovered_result': recovered_result,
            'strips_result': strips_result,
            'binarized_paths': binarized_paths,
            'final_deskewed_paths': final_deskewed_paths,
            'final_stage': final_stage,
            'final_outputs': final_outputs
        }
    
    def _process_deskew(
        self,
        image: np.ndarray,
        base_name: str,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> np.ndarray:
        """Process re-deskewing for fine-tuning."""
        if not self.config.enable_deskewing:
            if self.config.verbose:
                print(f"  Deskewing skipped (disabled)")
            return image
        
        deskew_start = time.time()
        
        # Perform fine-tuning deskew
        refined_deskewed, angle = self.deskew_processor.process(
            image,
            method=self.config.deskew_method,
            coarse_range=self.config.coarse_range,
            coarse_step=self.config.coarse_step,
            fine_range=self.config.fine_range,
            fine_step=self.config.fine_step,
            min_angle_correction=self.config.min_angle_correction,
        )
        
        # Save debug images if needed
        if self.config.save_debug_images and debug_dir:
            debug_subdir = debug_dir / "01_deskew" / base_name
            self.deskew_processor.save_debug_images_to_dir(debug_subdir)
        
        # Save refined deskewed if not in memory mode
        if not memory_mode:
            deskew_dir = self.config.output_dir / "01_refined_deskewed"
            deskew_path = deskew_dir / f"{base_name}_refined_deskewed.jpg"
            save_image(refined_deskewed, deskew_path)
        
        deskew_elapsed = time.time() - deskew_start
        print(f"    [TIMING] Deskewing ({self.config.deskew_method}): {deskew_elapsed:.3f} seconds")
        
        if self.config.verbose:
            if self.config.save_debug_images:
                print(f"  [DEBUG] Refined deskew: {angle:.2f} degrees (saving debug images)")
            else:
                print(f"  Refined deskew: {angle:.2f} degrees")
        
        return refined_deskewed
    
    def _process_table_detection(
        self,
        image: np.ndarray,
        base_name: str,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> tuple:
        """Process refined table line detection."""
        table_detect_start = time.time()
        
        # Detect table lines with refined parameters
        h_lines, v_lines = self.table_detector.process(
            image,
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
            search_region_top=self.config.search_region_top,
            search_region_bottom=self.config.search_region_bottom,
            search_region_left=self.config.search_region_left,
            search_region_right=self.config.search_region_right,
            skew_tolerance=getattr(self.config, 'skew_tolerance', 0),
            skew_angle_step=getattr(self.config, 'skew_angle_step', 0.2),
            # Line detection preprocessing parameters
            line_detection_use_preprocessing=self.config.line_detection_use_preprocessing,
            line_detection_binarization_method=self.config.line_detection_binarization_method,
            line_detection_binarization_threshold=self.config.line_detection_binarization_threshold,
            line_detection_adaptive_block_size=self.config.line_detection_adaptive_block_size,
            line_detection_adaptive_c=self.config.line_detection_adaptive_c,
            line_detection_binarization_invert=self.config.line_detection_binarization_invert,
            line_detection_binarization_denoise=self.config.line_detection_binarization_denoise,
            line_detection_stroke_enhancement=self.config.line_detection_stroke_enhancement,
            line_detection_stroke_kernel_size=self.config.line_detection_stroke_kernel_size,
            line_detection_stroke_iterations=self.config.line_detection_stroke_iterations,
            line_detection_stroke_kernel_shape=self.config.line_detection_stroke_kernel_shape,
        )
        
        # Save debug images if needed
        if self.config.save_debug_images and debug_dir:
            debug_subdir = debug_dir / "02_table_detection" / base_name
            self.table_detector.save_debug_images_to_dir(debug_subdir)
        
        # Save visualization if not in memory mode
        if not memory_mode:
            lines_dir = self.config.output_dir / "02_table_lines"
            lines_path = lines_dir / f"{base_name}_refined_table_lines.jpg"
            vis_image = visualize_detected_lines(image, h_lines, v_lines)
            save_image(vis_image, lines_path)
        
        # Always save line data JSON (needed for table recovery)
        lines_dir = self.config.output_dir / "02_table_lines"
        lines_data_dir = lines_dir / "lines_data"
        lines_data_dir.mkdir(parents=True, exist_ok=True)
        lines_json_path = lines_data_dir / f"{base_name}_lines_data.json"
        with open(lines_json_path, 'w') as f:
            json.dump({
                "horizontal_lines": [[int(x) for x in line] for line in h_lines],
                "vertical_lines": [[int(x) for x in line] for line in v_lines]
            }, f, indent=2)
        
        table_detect_elapsed = time.time() - table_detect_start
        print(f"    [TIMING] Table line detection: {table_detect_elapsed:.3f} seconds")
        
        if self.config.verbose:
            if self.config.save_debug_images:
                print(f"  [DEBUG] Table lines: {len(h_lines)} horizontal, {len(v_lines)} vertical (saving visualizations)")
            else:
                print(f"  Table lines: {len(h_lines)} horizontal, {len(v_lines)} vertical")
        
        return h_lines, v_lines, lines_json_path
    
    def _process_table_structure(
        self,
        image: np.ndarray,
        h_lines: List,
        v_lines: List,
        base_name: str,
        memory_mode: bool
    ) -> tuple:
        """Process table structure detection."""
        structure_start = time.time()
        
        table_structure, structure_analysis = detect_table_structure(
            h_lines,
            v_lines,
            eps=getattr(self.config, 'table_detection_eps', 10),
            return_analysis=True
        )
        
        # Save visualization if not in memory mode
        if not memory_mode:
            structure_dir = self.config.output_dir / "03_table_structure"
            structure_path = structure_dir / f"{base_name}_table_structure.jpg"
            structure_vis = visualize_table_structure(image, table_structure)
            save_image(structure_vis, structure_path)
        
        # Always save structure data JSON
        structure_dir = self.config.output_dir / "03_table_structure"
        structure_data_dir = structure_dir / "structure_data"
        structure_data_dir.mkdir(parents=True, exist_ok=True)
        structure_json_path = structure_data_dir / f"{base_name}_structure_data.json"
        with open(structure_json_path, 'w') as f:
            json.dump(table_structure, f, indent=2)
        
        structure_elapsed = time.time() - structure_start
        print(f"    [TIMING] Table structure detection: {structure_elapsed:.3f} seconds")
        
        if self.config.verbose:
            xs = table_structure.get("xs", [])
            ys = table_structure.get("ys", [])
            cells = table_structure.get("cells", [])
            if self.config.save_debug_images:
                print(f"  [DEBUG] Table structure: {len(cells)} cells in {len(xs)}x{len(ys)} grid (saving visualization)")
            else:
                print(f"  Table structure: {len(cells)} cells in {len(xs)}x{len(ys)} grid")
        
        return table_structure, structure_json_path
    
    def _process_table_recovery(
        self,
        image: np.ndarray,
        lines_json_path: Path,
        structure_json_path: Path,
        table_structure: Dict,
        base_name: str,
        memory_mode: bool
    ) -> Dict[str, Any]:
        """Process table recovery."""
        recovery_start = time.time()
        
        if table_structure.get("cells"):
            recovered_result = table_recovery(
                lines_json_path=str(lines_json_path),
                structure_json_path=str(structure_json_path),
                coverage_ratio=getattr(self.config, 'table_recovery_coverage_ratio', 0.8),
                background_image=image,
                highlight_merged=True,
                show_grid=True,
                label_cells=True
            )
            if self.config.verbose:
                recovered_cells = recovered_result.get('cells', [])
                if self.config.save_debug_images:
                    print(f"  [DEBUG] Table recovery: {len(recovered_cells)} cells recovered with merged cell detection")
                else:
                    print(f"  Table recovery: {len(recovered_cells)} cells recovered")
        else:
            # If no table structure detected, create a simple result
            recovered_result = {
                'visualization': image,
                'rows': [],
                'cols': [],
                'cells': []
            }
            if self.config.verbose:
                print(f"  Table recovery skipped (no table structure detected)")
        
        recovery_elapsed = time.time() - recovery_start
        print(f"    [TIMING] Table recovery: {recovery_elapsed:.3f} seconds")
        
        # Save recovered table visualization
        recovery_dir = self.config.output_dir / "04_table_recovered"
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
        
        if self.config.verbose:
            n_cells = len(recovered_result['cells'])
            n_merged = sum(1 for cell in recovered_result['cells'] 
                         if cell['rowspan'] > 1 or cell['colspan'] > 1)
            print(f"  Table recovered: {n_cells} cells ({n_merged} merged)")
            print(f"  Saved: {recovery_path.name}")
        
        return recovered_result
    
    def _process_vertical_strips(
        self,
        image: np.ndarray,
        recovered_result: Dict,
        base_name: str,
        memory_mode: bool
    ) -> Optional[Dict[str, Any]]:
        """Process vertical strip cutting."""
        # Check configuration
        vertical_strip_config = getattr(self.config, 'vertical_strip_cutting', {})
        if isinstance(vertical_strip_config, dict):
            # Support both old 'padding' and new 'padding_left'/'padding_right' format
            if 'padding' in vertical_strip_config:
                # Backward compatibility
                strip_padding_left = vertical_strip_config.get('padding', 20)
                strip_padding_right = vertical_strip_config.get('padding', 20)
            else:
                # New format with separate left/right padding
                strip_padding_left = vertical_strip_config.get('padding_left', 20)
                strip_padding_right = vertical_strip_config.get('padding_right', 20)
            strip_min_width = vertical_strip_config.get('min_width', 1)
            use_longest_lines_only = vertical_strip_config.get('use_longest_lines_only', False)
            min_length_ratio = vertical_strip_config.get('min_length_ratio', 0.9)
        else:
            strip_padding_left = 20
            strip_padding_right = 20
            strip_min_width = 1
            use_longest_lines_only = False
            min_length_ratio = 0.9
        
        if not self.config.enable_vertical_strip_cutting or not recovered_result.get("cells"):
            if self.config.verbose:
                print(f"  Vertical strip cutting skipped (disabled or no table structure)")
            return None
        
        strip_start = time.time()
        
        # Extract vertical lines from recovered table data
        recovered_v_lines = get_all_vertical_boundaries(recovered_result)
        
        # Check if we have enough vertical lines
        unique_x_positions = set()
        for x1, y1, x2, y2 in recovered_v_lines:
            unique_x_positions.add(x1)
        
        if len(unique_x_positions) < 2:
            if self.config.verbose:
                print(f"  Vertical strip cutting skipped (insufficient vertical boundaries: {len(unique_x_positions)} x-positions found, need at least 2)")
            strip_elapsed = time.time() - strip_start
            print(f"    [TIMING] Vertical strip cutting: {strip_elapsed:.3f} seconds")
            return None
        
        if self.config.verbose:
            print(f"  Extracted {len(recovered_v_lines)} vertical lines ({len(unique_x_positions)} unique x-positions)")
        
        strips_result = cut_vertical_strips(
            image=image,
            structure_json_path=None,
            padding_left=strip_padding_left,
            padding_right=strip_padding_right,
            min_width=strip_min_width,
            use_longest_lines_only=use_longest_lines_only,
            min_length_ratio=min_length_ratio,
            v_lines=recovered_v_lines,
            output_dir=self.config.output_dir / "05_vertical_strips",
            base_name=base_name,
            verbose=self.config.verbose
        )
        
        strip_elapsed = time.time() - strip_start
        print(f"    [TIMING] Vertical strip cutting: {strip_elapsed:.3f} seconds")
        
        if strips_result['num_strips'] > 0:
            if self.config.verbose:
                if self.config.save_debug_images:
                    print(f"  [DEBUG] Vertical strips: {strips_result['num_strips']} columns saved from {len(unique_x_positions)} boundaries")
                else:
                    print(f"  Vertical strips: {strips_result['num_strips']} columns saved")
            
            # Extract strip paths from the result for consistency
            strip_paths = []
            for strip_info in strips_result.get('strips', []):
                if 'path' in strip_info:
                    strip_paths.append(strip_info['path'])
            strips_result['strip_paths'] = strip_paths
        
        return strips_result
    
    def _process_binarization(
        self,
        strips_result: Optional[Dict],
        base_name: str,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> List[Path]:
        """Process binarization of vertical strips."""
        if not self.config.enable_binarization:
            if self.config.verbose:
                print(f"  Binarization skipped (disabled)")
            return []
        
        # Get strip files if available
        strips_dir = self.config.output_dir / "05_vertical_strips"
        strip_files = []
        if strips_dir.exists():
            strip_files = list(strips_dir.glob("*.png")) + list(strips_dir.glob("*.jpg"))
        
        if not strip_files:
            if self.config.verbose:
                print(f"  Binarization skipped (no vertical strips found)")
            return []
        
        binarized_dir = self.config.output_dir / "06_binarized"
        binarized_dir.mkdir(parents=True, exist_ok=True)
        binarized_paths = []
        
        # Process each vertical strip
        for i, strip_file in enumerate(strip_files):
            strip_img = load_image(strip_file)
            
            # Binarize the strip
            binarized_strip = self.binarize_processor.process(
                strip_img,
                method=self.config.binarization_method,
                threshold=self.config.binarization_threshold,
                adaptive_block_size=self.config.binarization_adaptive_block_size,
                adaptive_c=self.config.binarization_adaptive_c,
                invert=self.config.binarization_invert,
                denoise=self.config.binarization_denoise,
                enhance_strokes=self.config.stroke_enhancement_enable,
                stroke_kernel_size=self.config.stroke_enhancement_kernel_size,
                stroke_iterations=self.config.stroke_enhancement_iterations,
                stroke_kernel_shape=self.config.stroke_enhancement_kernel_shape,
            )
            
            # Save debug images for first strip only
            if self.config.save_debug_images and debug_dir and i == 0:
                debug_subdir = debug_dir / "06_binarization" / base_name
                self.binarize_processor.save_debug_images_to_dir(debug_subdir)
            
            # Save binarized strip
            binarized_strip_path = binarized_dir / strip_file.name
            save_image(binarized_strip, binarized_strip_path)
            binarized_paths.append(binarized_strip_path)
        
        if self.config.verbose:
            print(f"  Binarized {len(strip_files)} vertical strips using {self.config.binarization_method} method")
            print(f"  Saved to: {binarized_dir.name}")
        
        return binarized_paths
    
    def _process_final_deskew(
        self,
        binarized_paths: List[Path],
        strips_result: Optional[Dict],
        recovered_result: Dict,
        base_name: str,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> List[Path]:
        """Process final deskew as last optional step.
        
        Args:
            binarized_paths: Paths to binarized images if available
            strips_result: Vertical strips result if available
            recovered_result: Table recovery result
            base_name: Base name for output files
            memory_mode: Whether to use memory mode
            debug_dir: Optional debug directory
            
        Returns:
            List of paths to final deskewed images, or empty list if skipped
        """
        if not self.config.enable_final_deskew:
            if self.config.verbose:
                print(f"  Final deskew skipped (disabled)")
            return []
        
        final_deskew_start = time.time()
        
        # Determine which images to process
        images_to_process = []
        
        # Priority: binarized > vertical strips > recovered table
        if binarized_paths:
            images_to_process = [(path, load_image(path)) for path in binarized_paths]
            source_stage = "binarized"
        elif strips_result and strips_result.get('strip_paths'):
            strip_paths = strips_result['strip_paths']
            images_to_process = [(path, load_image(path)) for path in strip_paths]
            source_stage = "vertical strips"
        else:
            # Process the recovered table image
            recovered_path = self.config.output_dir / "04_table_recovered" / f"{base_name}_table_recovered.jpg"
            if recovered_path.exists():
                images_to_process = [(recovered_path, load_image(recovered_path))]
                source_stage = "recovered table"
            else:
                # If no recovered table file, use the visualization from memory
                if recovered_result and 'visualization' in recovered_result:
                    images_to_process = [(None, recovered_result['visualization'])]
                    source_stage = "recovered table (memory)"
                else:
                    if self.config.verbose:
                        print(f"  Final deskew skipped (no images to process)")
                    return []
        
        if self.config.verbose:
            print(f"  Processing final deskew on {len(images_to_process)} {source_stage} image(s)")
        
        # Create output directory
        final_deskewed_dir = self.config.output_dir / "07_final_deskewed"
        final_deskewed_dir.mkdir(parents=True, exist_ok=True)
        final_deskewed_paths = []
        
        # Process each image
        for i, (source_path, image) in enumerate(images_to_process):
            # Apply final deskew
            deskewed_image, angle = self.deskew_processor.process(
                image,
                method=self.config.final_deskew_method,
                use_binarization=self.config.final_deskew_use_binarization,
                coarse_range=self.config.final_deskew_coarse_range,
                coarse_step=self.config.final_deskew_coarse_step,
                fine_range=self.config.final_deskew_fine_range,
                fine_step=self.config.final_deskew_fine_step,
                min_angle_correction=self.config.final_deskew_min_angle_correction,
            )
            
            # Save debug images for first image only
            if self.config.save_debug_images and debug_dir and i == 0:
                debug_subdir = debug_dir / "07_final_deskew" / base_name
                self.deskew_processor.save_debug_images_to_dir(debug_subdir)
            
            # Determine output filename
            if source_path:
                output_name = source_path.name.replace('.jpg', '_final_deskewed.jpg').replace('.png', '_final_deskewed.png')
            else:
                output_name = f"{base_name}_final_deskewed.jpg"
            
            # Save processed image
            output_path = final_deskewed_dir / output_name
            save_image(deskewed_image, output_path)
            final_deskewed_paths.append(output_path)
            
            if self.config.verbose and i == 0:
                print(f"    Final deskew angle: {angle:.2f} degrees")
        
        final_deskew_elapsed = time.time() - final_deskew_start
        print(f"    [TIMING] Final deskew: {final_deskew_elapsed:.3f} seconds")
        
        if self.config.verbose:
            print(f"  Final deskew complete: {len(final_deskewed_paths)} images processed")
            print(f"  Saved to: {final_deskewed_dir.name}")
        
        return final_deskewed_paths