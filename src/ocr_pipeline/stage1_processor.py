"""Unified Stage 1 processor for all pipeline implementations."""

import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import json

from .config import Stage1Config
from .processors import (
    load_image,
    save_image,
    detect_table_lines,
    create_table_lines_mask,
    visualize_detected_lines,
    detect_table_structure,
    visualize_table_structure,
    TableDetectionProcessor,
    DeskewProcessor,
    PageSplitProcessor,
    MarginRemovalProcessor,
    MarkRemovalProcessor,
    TagRemovalProcessor,
    TableProcessingProcessor,
)


class Stage1Processor:
    """Unified processor for Stage 1 of the OCR pipeline.
    
    This class contains the complete Stage 1 processing logic that is shared
    across all pipeline implementations (regular, memory-efficient, and parallel).
    """
    
    def __init__(self, config: Stage1Config):
        """Initialize Stage 1 processor with configuration.
        
        Args:
            config: Stage 1 configuration
        """
        self.config = config
        
        # Initialize all processors
        self.table_detector = TableDetectionProcessor(config)
        self.deskew_processor = DeskewProcessor(config)
        self.margin_processor = MarginRemovalProcessor(config)
        self.page_split_processor = PageSplitProcessor(config)
        self.mark_removal_processor = MarkRemovalProcessor(config)
        self.tag_removal_processor = TagRemovalProcessor(config)
        self.table_processing_processor = TableProcessingProcessor(config)
    
    def process_image(
        self,
        image_path: Path,
        memory_mode: bool = False,
        debug_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Process a single image through Stage 1.
        
        Args:
            image_path: Path to input image
            memory_mode: If True, keeps results in memory instead of saving to disk
            debug_dir: Optional directory for debug images
            
        Returns:
            List of dictionaries containing processed results for each page
        """
        results = []
        
        # Show filename in verbose or debug mode
        if self.config.verbose or self.config.save_debug_images:
            print(f"\nProcessing: {image_path.name}")
        
        # Load image
        image = load_image(image_path)
        processing_image = image
        
        # Step 1: Mark removal (on full image)
        if self.config.enable_mark_removal:
            processing_image = self._process_mark_removal(
                processing_image, image_path, memory_mode, debug_dir
            )
        elif self.config.verbose:
            print(f"  Mark removal skipped (disabled)")
        
        # Step 2: Margin removal (on full image)
        if self.config.enable_margin_removal:
            processing_image = self._process_margin_removal(
                processing_image, image_path, memory_mode, debug_dir
            )
        elif self.config.verbose:
            print(f"  Margin removal skipped (disabled)")
        
        # Step 3: Deskewing (on full image - MOVED BEFORE PAGE SPLIT)
        if self.config.enable_deskewing:
            processing_image = self._process_deskew_full_image(
                processing_image, image_path, memory_mode, debug_dir
            )
        elif self.config.verbose:
            print(f"  Deskewing skipped (disabled)")
        
        # Step 4: Page splitting (on deskewed full image)
        if self.config.enable_page_splitting:
            pages = self._process_page_split(
                processing_image, image_path, memory_mode, debug_dir
            )
        else:
            # If page splitting is disabled, treat as single page
            pages = [("full", processing_image)]
            if self.config.verbose:
                print(f"  Page splitting skipped (disabled) - processing as single page")
        
        # Process each page
        for page_suffix, page_image in pages:
            page_name = f"{image_path.stem}_{page_suffix}"
            
            # Process the page and get results
            page_results = self._process_single_page(
                page_image, page_name, page_suffix, memory_mode, debug_dir
            )
            
            results.append(page_results)
        
        return results
    
    def _process_mark_removal(
        self,
        image: np.ndarray,
        image_path: Path,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> np.ndarray:
        """Process mark removal step."""
        table_lines_mask = None
        
        # If table line protection is enabled, detect table lines first
        if self.config.mark_removal_protect_table_lines:
            h_lines, v_lines = detect_table_lines(
                image,
                threshold=self.config.mark_removal_table_threshold,
                horizontal_kernel_size=self.config.mark_removal_table_h_kernel,
                vertical_kernel_size=self.config.mark_removal_table_v_kernel,
            )
            
            # Create mask from detected lines
            if h_lines or v_lines:
                table_lines_mask = create_table_lines_mask(
                    image.shape,
                    h_lines,
                    v_lines,
                    line_thickness=self.config.mark_removal_table_line_thickness
                )
                if self.config.verbose:
                    print(f"  Detected {len(h_lines)} horizontal and {len(v_lines)} vertical lines for protection")
        
        # Process mark removal
        processed = self.mark_removal_processor.process(
            image,
            dilate_iter=self.config.mark_removal_dilate_iter,
            kernel_size=self.config.mark_removal_kernel_size,
            table_lines_mask=table_lines_mask
        )
        
        # Save debug images if needed
        if self.config.save_debug_images and debug_dir:
            debug_subdir = debug_dir / "01_mark_removal" / image_path.stem
            self.mark_removal_processor.save_debug_images_to_dir(debug_subdir)
        
        # Save result if not in memory mode
        if not memory_mode:
            marks_dir = self.config.output_dir / "01_marks_removed"
            marks_path = marks_dir / f"{image_path.stem}_marks_removed.jpg"
            save_image(processed, marks_path)
        
        if self.config.verbose:
            if self.config.save_debug_images:
                print(f"  [DEBUG] Marks removed from full image (saving debug images)")
            else:
                print(f"  Marks removed from full image")
        
        return processed
    
    def _process_margin_removal(
        self,
        image: np.ndarray,
        image_path: Path,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> np.ndarray:
        """Process margin removal step."""
        margin_params = {
            'blur_ksize': getattr(self.config, 'margin_blur_ksize', 20),
            'close_ksize': getattr(self.config, 'margin_close_ksize', 30),
            'close_iter': getattr(self.config, 'margin_close_iter', 5),
            'erode_after_close': getattr(self.config, 'margin_erode_after_close', 0),
            'use_gradient_detection': getattr(self.config, 'margin_use_gradient_detection', False),
            'gradient_threshold': getattr(self.config, 'margin_gradient_threshold', 30),
        }
        
        processed = self.margin_processor.process(
            image,
            method="inscribed",
            **margin_params
        )
        
        # Save debug images if needed
        if self.config.save_debug_images and debug_dir:
            debug_subdir = debug_dir / "02_margin_removal" / image_path.stem
            self.margin_processor.save_debug_images_to_dir(debug_subdir)
        
        # Save result if not in memory mode
        if not memory_mode:
            margin_dir = self.config.output_dir / "02_margin_removed"
            margin_path = margin_dir / f"{image_path.stem}_margin_removed.jpg"
            save_image(processed, margin_path)
        
        if self.config.verbose:
            if self.config.save_debug_images:
                print(f"  [DEBUG] Margin removed from full image: {processed.shape} (saving debug images)")
            else:
                print(f"  Margin removed from full image: {processed.shape}")
        
        return processed
    
    def _process_deskew_full_image(
        self,
        image: np.ndarray,
        image_path: Path,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> np.ndarray:
        """Process deskewing on the full image (before page split)."""
        deskew_start = time.time()
        
        # Deskew the full image
        deskewed, angle = self.deskew_processor.process(
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
            debug_subdir = debug_dir / "03_deskew" / image_path.stem
            self.deskew_processor.save_debug_images_to_dir(debug_subdir)
        
        # Save result if not in memory mode
        if not memory_mode:
            deskew_dir = self.config.output_dir / "03_deskewed"
            deskew_path = deskew_dir / f"{image_path.stem}_deskewed.jpg"
            save_image(deskewed, deskew_path)
        
        deskew_elapsed = time.time() - deskew_start
        print(f"    [TIMING] Stage 1 Deskewing ({self.config.deskew_method}): {deskew_elapsed:.3f} seconds")
        
        if self.config.verbose:
            if self.config.save_debug_images:
                print(f"  [DEBUG] Full image deskewed: {angle:.2f} degrees using {self.config.deskew_method} method (saving debug images)")
            else:
                print(f"  Full image deskewed: {angle:.2f} degrees")
        
        return deskewed
    
    def _process_page_split(
        self,
        image: np.ndarray,
        image_path: Path,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> List[Tuple[str, np.ndarray]]:
        """Process page splitting step."""
        # Split the deskewed image
        right_page, left_page = self.page_split_processor.process(
            image,
            search_ratio=self.config.search_ratio,
            line_len_frac=getattr(self.config, 'line_len_frac', 0.3),
            line_thick=getattr(self.config, 'line_thick', 3),
            peak_thr=getattr(self.config, 'peak_thr', 0.3),
        )
        
        # Save debug images if needed
        if self.config.save_debug_images and debug_dir:
            debug_subdir = debug_dir / "04_page_split" / image_path.stem
            self.page_split_processor.save_debug_images_to_dir(debug_subdir)
        
        # Save split pages if not in memory mode
        if not memory_mode:
            split_dir = self.config.output_dir / "04_split_pages"
            for page_name, page in [("left", left_page), ("right", right_page)]:
                split_path = split_dir / f"{image_path.stem}_{page_name}.jpg"
                save_image(page, split_path)
                if self.config.verbose:
                    print(f"  Split page saved: {split_path.name}")
        
        return [("left", left_page), ("right", right_page)]
    
    def _process_single_page(
        self,
        page_image: np.ndarray,
        page_name: str,
        page_suffix: str,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> Dict[str, Any]:
        """Process a single page through remaining Stage 1 steps.
        
        Args:
            page_image: The page image to process
            page_name: Full name of the page (e.g., "image_left")
            page_suffix: Page suffix (e.g., "left", "right", "full")
            memory_mode: Whether to use memory mode
            debug_dir: Optional debug directory
            
        Returns:
            Dictionary containing processed results
        """
        processing_image = page_image
        
        # Step 4b: Tag removal (per page)
        if self.config.enable_tag_removal:
            processing_image = self._process_tag_removal(
                processing_image, page_name, memory_mode, debug_dir
            )
        elif self.config.verbose:
            print(f"    Tag removal skipped (disabled): {page_name}")
        
        # Step 5: Table line detection
        h_lines, v_lines = self._process_table_detection(
            processing_image, page_name, memory_mode, debug_dir
        )
        
        # Step 6: Table structure detection
        table_structure = self._process_table_structure(
            processing_image, h_lines, v_lines, page_name, memory_mode
        )
        
        # Step 7: Table cropping
        cropped_table = self._process_table_cropping(
            processing_image, table_structure, page_name, memory_mode, debug_dir
        )
        
        # Return results
        return {
            'page_name': page_name,
            'page_suffix': page_suffix,
            'image': cropped_table,
            'h_lines': h_lines,
            'v_lines': v_lines,
            'table_structure': table_structure,
            'cropped_path': self.config.output_dir / "07_border_cropped" / f"{page_name}_border_cropped.jpg"
        }
    
    def _process_tag_removal(
        self,
        image: np.ndarray,
        page_name: str,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> np.ndarray:
        """Process tag removal step."""
        processed = self.tag_removal_processor.process(
            image,
            method=self.config.tag_removal_method,
            thresh_dark=self.config.tag_removal_thresh_dark,
            row_sum_thresh=self.config.tag_removal_row_sum_thresh,
            dark_ratio=self.config.tag_removal_dark_ratio,
            min_area=self.config.tag_removal_min_area,
            max_area=self.config.tag_removal_max_area,
            min_aspect=self.config.tag_removal_min_aspect,
            max_aspect=self.config.tag_removal_max_aspect,
            band_top=self.config.tag_removal_band_top,
            band_bottom=self.config.tag_removal_band_bottom,
            rows_mode=self.config.tag_removal_rows_mode,
            min_dark=self.config.tag_removal_min_dark,
            min_score=self.config.tag_removal_min_score,
            reject_red=self.config.tag_removal_reject_red,
            nms_iou=self.config.tag_removal_nms_iou,
            pad_px=self.config.tag_removal_pad_px,
            min_width_ratio=self.config.tag_removal_min_width_ratio,
            max_width_ratio=self.config.tag_removal_max_width_ratio,
            min_height_ratio=self.config.tag_removal_min_height_ratio,
            max_height_ratio=self.config.tag_removal_max_height_ratio,
            min_aspect_ratio=self.config.tag_removal_min_aspect_ratio,
            max_aspect_ratio=self.config.tag_removal_max_aspect_ratio,
            glyph_kernel_size=self.config.tag_removal_glyph_kernel_size,
            mask_close_kernel_size=self.config.tag_removal_mask_close_kernel_size,
            mask_open_kernel_size=self.config.tag_removal_mask_open_kernel_size,
            mask_dilate_kernel_size=self.config.tag_removal_mask_dilate_kernel_size,
        )
        
        # Save debug images if needed
        if self.config.save_debug_images and debug_dir:
            debug_subdir = debug_dir / "04b_tag_removal" / page_name
            self.tag_removal_processor.save_debug_images_to_dir(debug_subdir)
        
        # Save result if not in memory mode
        if not memory_mode:
            tags_dir = self.config.output_dir / "04b_tags_removed"
            tags_path = tags_dir / f"{page_name}_tags_removed.jpg"
            save_image(processed, tags_path)
        
        if self.config.verbose:
            if self.config.save_debug_images:
                print(f"    [DEBUG] Generation tags removed: {page_name} (saving debug images)")
            else:
                print(f"    Generation tags removed: {page_name}")
        
        return processed
    
    def _process_table_detection(
        self,
        image: np.ndarray,
        page_name: str,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> Tuple[List, List]:
        """Process table line detection step."""
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
        )
        
        # Save debug images if needed
        if self.config.save_debug_images and debug_dir:
            debug_subdir = debug_dir / "05_table_detection" / page_name
            self.table_detector.save_debug_images_to_dir(debug_subdir)
        
        # Save visualization if not in memory mode
        if not memory_mode:
            lines_dir = self.config.output_dir / "05_table_lines"
            lines_path = lines_dir / f"{page_name}_table_lines.jpg"
            vis_image = visualize_detected_lines(image, h_lines, v_lines)
            save_image(vis_image, lines_path)
        
        # Always save lines data JSON (needed for next steps)
        lines_dir = self.config.output_dir / "05_table_lines"
        lines_data_dir = lines_dir / "lines_data"
        lines_data_dir.mkdir(parents=True, exist_ok=True)
        lines_json_path = lines_data_dir / f"{page_name}_lines_data.json"
        lines_data = {
            "horizontal_lines": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in h_lines],
            "vertical_lines": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in v_lines]
        }
        with open(lines_json_path, 'w') as f:
            json.dump(lines_data, f, indent=2)
        
        if self.config.verbose:
            if self.config.save_debug_images:
                print(f"  [DEBUG] Table lines: {len(h_lines)} horizontal, {len(v_lines)} vertical (saving visualizations)")
            else:
                print(f"  Table lines: {len(h_lines)} horizontal, {len(v_lines)} vertical")
        
        return h_lines, v_lines
    
    def _process_table_structure(
        self,
        image: np.ndarray,
        h_lines: List,
        v_lines: List,
        page_name: str,
        memory_mode: bool
    ) -> Dict[str, Any]:
        """Process table structure detection step."""
        table_structure, structure_analysis = detect_table_structure(
            h_lines,
            v_lines,
            eps=getattr(self.config, 'table_detection_eps', 10),
            return_analysis=True
        )
        
        # Save visualization if not in memory mode
        if not memory_mode and table_structure.get("cells"):
            structure_dir = self.config.output_dir / "06_table_structure"
            structure_path = structure_dir / f"{page_name}_table_structure.jpg"
            structure_vis = visualize_table_structure(image, table_structure)
            save_image(structure_vis, structure_path)
        
        # Always save structure data JSON (needed for next steps)
        structure_dir = self.config.output_dir / "06_table_structure"
        structure_data_dir = structure_dir / "structure_data"
        structure_data_dir.mkdir(parents=True, exist_ok=True)
        structure_json_path = structure_data_dir / f"{page_name}_structure_data.json"
        with open(structure_json_path, 'w') as f:
            json.dump(table_structure, f, indent=2)
        
        if self.config.verbose:
            xs = table_structure.get("xs", [])
            ys = table_structure.get("ys", [])
            cells = table_structure.get("cells", [])
            if self.config.save_debug_images:
                print(f"  [DEBUG] Table structure: {len(cells)} cells in {len(xs)}x{len(ys)} grid (saving visualization)")
            else:
                print(f"  Table structure: {len(cells)} cells in {len(xs)}x{len(ys)} grid")
        
        return table_structure
    
    def _process_table_cropping(
        self,
        image: np.ndarray,
        table_structure: Dict[str, Any],
        page_name: str,
        memory_mode: bool,
        debug_dir: Optional[Path]
    ) -> np.ndarray:
        """Process table cropping step."""
        if self.config.enable_table_cropping and table_structure.get("cells"):
            cropped = self.table_processing_processor.process(
                image,
                table_structure,
                operation="crop",
                padding=getattr(self.config, 'table_crop_padding', 20),
                return_analysis=False
            )
            
            # Save debug images if needed
            if self.config.save_debug_images and debug_dir:
                debug_subdir = debug_dir / "06_table_cropping" / page_name
                self.table_processing_processor.save_debug_images_to_dir(debug_subdir)
        else:
            # If cropping is disabled or no table structure, use original image
            cropped = image
            if self.config.verbose:
                if not self.config.enable_table_cropping:
                    print(f"  Table cropping skipped (disabled)")
                else:
                    print(f"  Table cropping skipped (no table structure detected)")
        
        # Save cropped table
        crop_dir = self.config.output_dir / "07_border_cropped"
        crop_path = crop_dir / f"{page_name}_border_cropped.jpg"
        save_image(cropped, crop_path)
        
        if self.config.verbose:
            print(f"  Table cropped: {crop_path.name}")
        
        return cropped