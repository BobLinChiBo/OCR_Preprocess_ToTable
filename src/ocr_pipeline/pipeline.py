"""Main OCR pipeline."""

import argparse
import json
from pathlib import Path
from typing import List
from datetime import datetime

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
    TableProcessingProcessor,
    BinarizeProcessor,
)
from .processors.table_recovery_utils import get_major_vertical_boundaries


class OCRPipeline:
    """Simple OCR table extraction pipeline."""

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
                self.config.angle_range,
                self.config.angle_step,
                self.config.min_angle_correction,
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

        print(f"Processing complete. {len(all_outputs)} files created.")
        return all_outputs


class TwoStageOCRPipeline:
    """Two-stage OCR pipeline for professional table extraction."""

    def __init__(
        self, stage1_config: Stage1Config = None, stage2_config: Stage2Config = None
    ):
        """Initialize two-stage pipeline with configurations."""
        self.stage1_config = stage1_config or get_stage1_config()
        self.stage2_config = stage2_config or get_stage2_config()

        # Lazy initialization - pipelines created only when needed
        self._stage1_pipeline = None
        self._stage2_pipeline = None
        
        # Initialize processors for debug mode
        self.table_detector_s1 = TableDetectionProcessor(self.stage1_config)
        self.table_detector_s2 = TableDetectionProcessor(self.stage2_config)
        self.deskew_processor_s1 = DeskewProcessor(self.stage1_config)
        self.deskew_processor_s2 = DeskewProcessor(self.stage2_config)
        self.margin_processor_s1 = MarginRemovalProcessor(self.stage1_config)
        self.margin_processor_s2 = MarginRemovalProcessor(self.stage2_config)
        self.page_split_processor_s1 = PageSplitProcessor(self.stage1_config)
        self.mark_removal_processor_s1 = MarkRemovalProcessor(self.stage1_config)
        self.mark_removal_processor_s2 = MarkRemovalProcessor(self.stage2_config)
        self.table_processing_processor_s1 = TableProcessingProcessor(self.stage1_config)
        self.table_processing_processor_s2 = TableProcessingProcessor(self.stage2_config)
        self.binarize_processor_s2 = BinarizeProcessor(self.stage2_config)
        
        # Generate single timestamp for entire pipeline run
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.debug_run_dir_s1 = None
        self.debug_run_dir_s2 = None
        
        # Create debug directories if debug mode is enabled
        if self.stage1_config.save_debug_images and self.stage1_config.debug_dir:
            self.debug_run_dir_s1 = self.stage1_config.debug_dir / f"{self.run_timestamp}_run"
            self.debug_run_dir_s1.mkdir(parents=True, exist_ok=True)
            
        if self.stage2_config.save_debug_images and self.stage2_config.debug_dir:
            self.debug_run_dir_s2 = self.stage2_config.debug_dir / f"{self.run_timestamp}_run"
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
                "debug_dir": str(debug_dir.parent),
                "verbose": self.stage1_config.verbose if stage == "stage1" else self.stage2_config.verbose
            }
        }
        
        info_file = debug_dir / "run_info.json"
        with open(info_file, "w") as f:
            json.dump(run_info, f, indent=2)

    @property
    def stage1_pipeline(self) -> OCRPipeline:
        """Get or create Stage 1 pipeline on demand."""
        if self._stage1_pipeline is None:
            self._stage1_pipeline = OCRPipeline(self.stage1_config)
        return self._stage1_pipeline

    @property
    def stage2_pipeline(self) -> OCRPipeline:
        """Get or create Stage 2 pipeline on demand."""
        if self._stage2_pipeline is None:
            self._stage2_pipeline = OCRPipeline(self.stage2_config)
        return self._stage2_pipeline

    def run_stage1(self, input_path: Path = None) -> List[Path]:
        """
        Run Stage 1: Initial processing and table cropping.

        Args:
            input_path: Input directory or single image file (optional)

        Returns:
            List of cropped table image paths ready for Stage 2
        """
        if self.stage1_config.verbose:
            print("*** STARTING STAGE 1: INITIAL PROCESSING ***")
            print("=" * 50)

        # Process images through complete Stage 1 pipeline
        input_path = input_path or self.stage1_config.input_dir
        cropped_tables = []

        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        # Handle single file or directory
        if input_path.is_file():
            # Single file processing
            image_files = [input_path]
        else:
            # Directory processing
            image_files = get_image_files(input_path)
            if not image_files:
                raise ValueError(f"No image files found in: {input_path}")

        if self.stage1_config.verbose:
            print(f"Found {len(image_files)} images to process")

        # Save run info at start
        self._save_run_info("stage1", input_path, len(image_files), "started")

        # Process each image through Stage 1 steps
        for image_path in image_files:
            try:
                if self.stage1_config.verbose:
                    print(f"\nProcessing: {image_path.name}")

                # Load image
                image = load_image(image_path)
                processing_image = image

                # Step 1: Mark removal (on full image)
                if self.stage1_config.enable_mark_removal:
                    table_lines_mask = None
                    
                    # If table line protection is enabled, detect table lines first
                    if self.stage1_config.mark_removal_protect_table_lines:
                        h_lines, v_lines = detect_table_lines(
                            processing_image,
                            threshold=self.stage1_config.mark_removal_table_threshold,
                            horizontal_kernel_size=self.stage1_config.mark_removal_table_h_kernel,
                            vertical_kernel_size=self.stage1_config.mark_removal_table_v_kernel,
                        )
                        
                        # Create mask from detected lines
                        if h_lines or v_lines:
                            table_lines_mask = create_table_lines_mask(
                                processing_image.shape,
                                h_lines,
                                v_lines,
                                line_thickness=self.stage1_config.mark_removal_table_line_thickness
                            )
                            if self.stage1_config.verbose:
                                print(f"  Detected {len(h_lines)} horizontal and {len(v_lines)} vertical lines for protection")
                    
                    if self.stage1_config.save_debug_images:
                        # Use processor for debug mode
                        processing_image = self.mark_removal_processor_s1.process(
                            processing_image,
                            dilate_iter=self.stage1_config.mark_removal_dilate_iter,
                            kernel_size=self.stage1_config.mark_removal_kernel_size,
                            table_lines_mask=table_lines_mask
                        )
                        
                        # Save debug images if debug directory is configured
                        if self.debug_run_dir_s1:
                            debug_subdir = self.debug_run_dir_s1 / "01_mark_removal" / image_path.stem
                            self.mark_removal_processor_s1.save_debug_images_to_dir(debug_subdir)
                    else:
                        # Use direct function call for normal processing
                        processing_image = remove_marks(
                            processing_image,
                            dilate_iter=self.stage1_config.mark_removal_dilate_iter,
                            kernel_size=self.stage1_config.mark_removal_kernel_size,
                            table_lines_mask=table_lines_mask
                        )
                    
                    # Save marks-removed image
                    if self.stage1_config.save_intermediate_outputs:
                        marks_dir = self.stage1_config.output_dir / "01_marks_removed"
                        marks_path = marks_dir / f"{image_path.stem}_marks_removed.jpg"
                        save_image(processing_image, marks_path)
                    
                    if self.stage1_config.verbose:
                        print(f"  Marks removed from full image")
                elif self.stage1_config.verbose:
                    print(f"  Mark removal skipped (disabled)")

                # Step 2: Margin removal (on cleaned full image)
                if self.stage1_config.enable_margin_removal:
                    # Get margin removal parameters from config
                    margin_params = {
                        'blur_ksize': getattr(self.stage1_config, 'margin_blur_ksize', 20),
                        'close_ksize': getattr(self.stage1_config, 'margin_close_ksize', 30),
                        'close_iter': getattr(self.stage1_config, 'margin_close_iter', 5),
                        'erode_after_close': getattr(self.stage1_config, 'margin_erode_after_close', 0),
                        'use_gradient_detection': getattr(self.stage1_config, 'margin_use_gradient_detection', False),
                        'gradient_threshold': getattr(self.stage1_config, 'margin_gradient_threshold', 30),
                    }
                    
                    if self.stage1_config.save_debug_images:
                        # Use processor for debug mode
                        processing_image = self.margin_processor_s1.process(
                            processing_image,
                            method="inscribed",
                            **margin_params
                        )
                        
                        # Save debug images if debug directory is configured
                        if self.debug_run_dir_s1:
                            debug_subdir = self.debug_run_dir_s1 / "02_margin_removal" / image_path.stem
                            self.margin_processor_s1.save_debug_images_to_dir(debug_subdir)
                    else:
                        # Use direct function call for normal processing
                        processing_image = remove_margin_inscribed(
                            processing_image,
                            **margin_params
                        )
                    
                    # Save margin-removed image
                    if self.stage1_config.save_intermediate_outputs:
                        margin_dir = self.stage1_config.output_dir / "02_margin_removed"
                        margin_path = margin_dir / f"{image_path.stem}_margin_removed.jpg"
                        save_image(processing_image, margin_path)
                    
                    if self.stage1_config.verbose:
                        print(f"  Margin removed from full image: {processing_image.shape}")
                elif self.stage1_config.verbose:
                    print(f"  Margin removal skipped (disabled)")

                # Step 3: Split processed image into pages
                if self.stage1_config.enable_page_splitting:
                    if self.stage1_config.save_debug_images:
                        # Use processor for debug mode
                        right_page, left_page = self.page_split_processor_s1.process(
                            processing_image,
                            search_ratio=self.stage1_config.search_ratio,
                            line_len_frac=getattr(self.stage1_config, 'line_len_frac', 0.3),
                            line_thick=getattr(self.stage1_config, 'line_thick', 3),
                            peak_thr=getattr(self.stage1_config, 'peak_thr', 0.3),
                        )
                        
                        # Save debug images if debug directory is configured
                        if self.debug_run_dir_s1:
                            debug_subdir = self.debug_run_dir_s1 / "03_page_split" / image_path.stem
                            self.page_split_processor_s1.save_debug_images_to_dir(debug_subdir)
                    else:
                        # Use direct function call for normal processing
                        right_page, left_page = split_two_page_image(
                            processing_image,
                            search_ratio=self.stage1_config.search_ratio,
                            line_len_frac=getattr(self.stage1_config, 'line_len_frac', 0.3),
                            line_thick=getattr(self.stage1_config, 'line_thick', 3),
                            peak_thr=getattr(self.stage1_config, 'peak_thr', 0.3),
                        )
                else:
                    # If page splitting is disabled, treat the whole image as a single page
                    right_page = None
                    left_page = processing_image
                    if self.stage1_config.verbose:
                        print(f"  Page splitting skipped (disabled) - processing as single page")

                # Save split pages
                if self.stage1_config.save_intermediate_outputs and self.stage1_config.enable_page_splitting:
                    split_dir = self.stage1_config.output_dir / "03_split_pages"
                    for page_name, page in [("left", left_page), ("right", right_page)]:
                        split_path = split_dir / f"{image_path.stem}_{page_name}.jpg"
                        save_image(page, split_path)
                        if self.stage1_config.verbose:
                            print(f"  Split page saved: {split_path.name}")

                # Process each split page
                if self.stage1_config.enable_page_splitting:
                    pages_to_process = [("left", left_page), ("right", right_page)]
                else:
                    pages_to_process = [("full", left_page)]  # left_page contains the full image when splitting is disabled
                
                for page_suffix, page in pages_to_process:
                    page_name = f"{image_path.stem}_{page_suffix}"
                    processing_image = page

                    # Step 4: Deskew (per page)
                    if self.stage1_config.enable_deskewing:
                        if self.stage1_config.save_debug_images:
                            # Use processor for debug mode
                            deskewed, angle = self.deskew_processor_s1.process(
                                processing_image,
                                angle_range=self.stage1_config.angle_range,
                                angle_step=self.stage1_config.angle_step,
                                min_angle_correction=self.stage1_config.min_angle_correction,
                            )
                            
                            # Save debug images if debug directory is configured
                            if self.debug_run_dir_s1:
                                debug_subdir = self.debug_run_dir_s1 / "04_deskew" / page_name
                                self.deskew_processor_s1.save_debug_images_to_dir(debug_subdir)
                        else:
                            # Use direct function call for normal processing
                            deskewed, angle = deskew_image(
                                processing_image,
                                self.stage1_config.angle_range,
                                self.stage1_config.angle_step,
                                self.stage1_config.min_angle_correction,
                            )

                        # Save deskewed image
                        if self.stage1_config.save_intermediate_outputs:
                            deskew_dir = self.stage1_config.output_dir / "04_deskewed"
                            deskew_path = deskew_dir / f"{page_name}_deskewed.jpg"
                            save_image(deskewed, deskew_path)

                        if self.stage1_config.verbose:
                            print(f"  Deskewed: {angle:.2f} degrees")
                    else:
                        deskewed = processing_image
                        if self.stage1_config.verbose:
                            print(f"  Deskewing skipped (disabled)")

                    # Step 5: Table line detection (always enabled - generates required JSON)
                    if self.stage1_config.save_debug_images:
                        # Use processor for debug mode
                        h_lines, v_lines = self.table_detector_s1.process(
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
                            skew_tolerance=getattr(self.stage1_config, 'skew_tolerance', 0),
                            skew_angle_step=getattr(self.stage1_config, 'skew_angle_step', 0.2),
                        )
                        
                        # Save debug images if debug directory is configured
                        if self.debug_run_dir_s1:
                            debug_subdir = self.debug_run_dir_s1 / "05_table_detection" / page_name
                            self.table_detector_s1.save_debug_images_to_dir(debug_subdir)
                    else:
                        # Use direct function call for normal processing
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
                            skew_tolerance=getattr(self.stage1_config, 'skew_tolerance', 0),
                            skew_angle_step=getattr(self.stage1_config, 'skew_angle_step', 0.2),
                        )

                    # Save table line visualization
                    if self.stage1_config.save_intermediate_outputs or self.stage1_config.save_debug_images:
                        lines_dir = self.stage1_config.output_dir / "05_table_lines"
                        lines_path = lines_dir / f"{page_name}_table_lines.jpg"
                        vis_image = visualize_detected_lines(deskewed, h_lines, v_lines)
                        save_image(vis_image, lines_path)

                    if self.stage1_config.verbose:
                        print(f"  Table lines: {len(h_lines)} horizontal, {len(v_lines)} vertical")

                    # Save lines data to JSON for next step
                    lines_dir = self.stage1_config.output_dir / "05_table_lines"
                    lines_data_dir = lines_dir / "lines_data"
                    lines_data_dir.mkdir(parents=True, exist_ok=True)
                    lines_json_path = lines_data_dir / f"{page_name}_lines_data.json"
                    lines_data = {
                        "horizontal_lines": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in h_lines],
                        "vertical_lines": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in v_lines]
                    }
                    with open(lines_json_path, 'w') as f:
                        json.dump(lines_data, f, indent=2)

                    # Step 6: Table structure detection (always enabled - generates required JSON)
                    table_structure, structure_analysis = detect_table_structure(
                        h_lines,  # Pass horizontal lines
                        v_lines,  # Pass vertical lines
                        eps=getattr(self.stage1_config, 'table_detection_eps', 10),
                        return_analysis=True
                    )

                    # Save table structure visualization
                    if (self.stage1_config.save_intermediate_outputs or self.stage1_config.save_debug_images) and table_structure.get("cells"):
                        structure_dir = self.stage1_config.output_dir / "06_table_structure"
                        structure_path = structure_dir / f"{page_name}_table_structure.jpg"
                        structure_vis = visualize_table_structure(deskewed, table_structure)
                        save_image(structure_vis, structure_path)

                    if self.stage1_config.verbose:
                        xs = table_structure.get("xs", [])
                        ys = table_structure.get("ys", [])
                        cells = table_structure.get("cells", [])
                        print(f"  Table structure: {len(cells)} cells in {len(xs)}x{len(ys)} grid")

                    # Save table structure data to JSON for next step
                    structure_dir = self.stage1_config.output_dir / "06_table_structure"
                    structure_data_dir = structure_dir / "structure_data"
                    structure_data_dir.mkdir(parents=True, exist_ok=True)
                    structure_json_path = structure_data_dir / f"{page_name}_structure_data.json"
                    with open(structure_json_path, 'w') as f:
                        json.dump(table_structure, f, indent=2)

                    # Step 7: Crop deskewed image using detected table borders with padding
                    if self.stage1_config.enable_table_cropping and table_structure.get("cells"):
                        if self.stage1_config.save_debug_images:
                            # Use processor for debug mode
                            cropped_table = self.table_processing_processor_s1.process(
                                deskewed,
                                table_structure,
                                operation="crop",
                                padding=getattr(self.stage1_config, 'table_crop_padding', 20),
                                return_analysis=False
                            )
                            
                            # Save debug images if debug directory is configured
                            if self.debug_run_dir_s1:
                                debug_subdir = self.debug_run_dir_s1 / "06_table_cropping" / page_name
                                self.table_processing_processor_s1.save_debug_images_to_dir(debug_subdir)
                        else:
                            # Use direct function call for normal processing
                            cropped_table = crop_to_table_borders(
                                deskewed, 
                                table_structure,
                                padding=getattr(self.stage1_config, 'table_crop_padding', 20),
                                return_analysis=False
                            )
                    else:
                        # If cropping is disabled or no table structure detected, use the deskewed image as-is
                        cropped_table = deskewed
                        if self.stage1_config.verbose:
                            if not self.stage1_config.enable_table_cropping:
                                print(f"  Table cropping skipped (disabled)")
                            else:
                                print(f"  Table cropping skipped (no table structure detected)")

                    # Save border-cropped table for Stage 2
                    crop_dir = self.stage1_config.output_dir / "07_border_cropped"
                    crop_path = crop_dir / f"{page_name}_border_cropped.jpg"
                    save_image(cropped_table, crop_path)
                    cropped_tables.append(crop_path)

                    if self.stage1_config.verbose:
                        print(f"  Table cropped: {crop_path.name}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if self.stage1_config.verbose:
            print(
                f"\n*** STAGE 1 COMPLETE: {len(cropped_tables)} cropped tables ready for Stage 2 ***"
            )
            print(f"Output: {self.stage1_config.output_dir / '07_border_cropped'}")

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

        refined_tables = []

        # Process each cropped table through Stage 2 refinement
        for image_path in image_files:
            try:
                if self.stage2_config.verbose:
                    print(f"\nRefining: {image_path.name}")

                # Load cropped table
                table_image = load_image(image_path)
                base_name = image_path.stem.replace("_cropped", "")

                # Re-deskew for fine-tuning
                if self.stage2_config.enable_deskewing:
                    if self.stage2_config.save_debug_images:
                        # Use processor for debug mode
                        refined_deskewed, _ = self.deskew_processor_s2.process(
                            table_image,
                            angle_range=self.stage2_config.angle_range,
                            angle_step=self.stage2_config.angle_step,
                            min_angle_correction=self.stage2_config.min_angle_correction,
                        )
                        
                        # Save debug images if debug directory is configured
                        if self.debug_run_dir_s2:
                            debug_subdir = self.debug_run_dir_s2 / "01_deskew" / base_name
                            self.deskew_processor_s2.save_debug_images_to_dir(debug_subdir)
                    else:
                        # Use direct function call for normal processing
                        refined_deskewed, _ = deskew_image(
                            table_image,
                            self.stage2_config.angle_range,
                            self.stage2_config.angle_step,
                            self.stage2_config.min_angle_correction,
                        )

                    # Save refined deskewed
                    if self.stage2_config.save_intermediate_outputs:
                        deskew_dir = self.stage2_config.output_dir / "01_deskewed"
                        deskew_path = deskew_dir / f"{base_name}_refined_deskewed.jpg"
                        save_image(refined_deskewed, deskew_path)
                else:
                    refined_deskewed = table_image
                    if self.stage2_config.verbose:
                        print(f"  Deskewing skipped (disabled)")

                # Refined line detection (always enabled - generates required JSON)
                if self.stage2_config.save_debug_images:
                    # Use processor for debug mode
                    h_lines, v_lines = self.table_detector_s2.process(
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
                        skew_tolerance=getattr(self.stage2_config, 'skew_tolerance', 0),
                        skew_angle_step=getattr(self.stage2_config, 'skew_angle_step', 0.2),
                    )
                    
                    # Save debug images if debug directory is configured
                    if self.debug_run_dir_s2:
                        debug_subdir = self.debug_run_dir_s2 / "02_table_detection" / base_name
                        self.table_detector_s2.save_debug_images_to_dir(debug_subdir)
                else:
                    # Use direct function call for normal processing
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
                        skew_tolerance=getattr(self.stage2_config, 'skew_tolerance', 0),
                        skew_angle_step=getattr(self.stage2_config, 'skew_angle_step', 0.2),
                    )

                # Save refined line detection visualization
                if self.stage2_config.save_intermediate_outputs or self.stage2_config.save_debug_images:
                    lines_dir = self.stage2_config.output_dir / "02_table_lines"
                    lines_path = lines_dir / f"{base_name}_refined_table_lines.jpg"
                    vis_image = visualize_detected_lines(refined_deskewed, h_lines, v_lines)
                    save_image(vis_image, lines_path)
                
                # Save line data to JSON for table recovery
                lines_dir = self.stage2_config.output_dir / "02_table_lines"
                lines_data_dir = lines_dir / "lines_data"
                lines_data_dir.mkdir(parents=True, exist_ok=True)
                lines_json_path = lines_data_dir / f"{base_name}_lines_data.json"
                with open(lines_json_path, 'w') as f:
                    json.dump({
                        "horizontal_lines": [[int(x) for x in line] for line in h_lines],
                        "vertical_lines": [[int(x) for x in line] for line in v_lines]
                    }, f, indent=2)

                if self.stage2_config.verbose:
                    print(f"  Table lines: {len(h_lines)} horizontal, {len(v_lines)} vertical")

                # Step 3: Table structure detection (always enabled - generates required JSON)
                table_structure, structure_analysis = detect_table_structure(
                    h_lines,  # Pass horizontal lines
                    v_lines,  # Pass vertical lines
                    eps=getattr(self.stage2_config, 'table_detection_eps', 10),
                    return_analysis=True
                )

                # Save table structure visualization
                if self.stage2_config.save_intermediate_outputs or self.stage2_config.save_debug_images:
                    structure_dir = self.stage2_config.output_dir / "03_table_structure"
                    structure_path = structure_dir / f"{base_name}_table_structure.jpg"
                    structure_vis = visualize_table_structure(refined_deskewed, table_structure)
                    save_image(structure_vis, structure_path)

                if self.stage2_config.verbose:
                    xs = table_structure.get("xs", [])
                    ys = table_structure.get("ys", [])
                    cells = table_structure.get("cells", [])
                    print(f"  Table structure: {len(cells)} cells in {len(xs)}x{len(ys)} grid")

                # Save table structure data to JSON for reference
                structure_dir = self.stage2_config.output_dir / "03_table_structure"
                structure_data_dir = structure_dir / "structure_data"
                structure_data_dir.mkdir(parents=True, exist_ok=True)
                structure_json_path = structure_data_dir / f"{base_name}_structure_data.json"
                with open(structure_json_path, 'w') as f:
                    json.dump(table_structure, f, indent=2)

                # Step 4: Table recovery (always enabled - generates required JSON)
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
                    # If no table structure detected, create a simple result
                    recovered_result = {
                        'visualization': refined_deskewed,
                        'rows': [],
                        'cols': [],
                        'cells': []
                    }
                    if self.stage2_config.verbose:
                        print(f"  Table recovery skipped (no table structure detected)")

                # Save recovered table visualization as final result
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
                
                refined_tables.append(recovery_path)

                if self.stage2_config.verbose:
                    n_cells = len(recovered_result['cells'])
                    n_merged = sum(1 for cell in recovered_result['cells'] 
                                 if cell['rowspan'] > 1 or cell['colspan'] > 1)
                    print(f"  Table recovered: {n_cells} cells ({n_merged} merged)")
                    print(f"  Saved: {recovery_path.name}")

                # Step 5: Cut vertical strips (optional)
                # Check for vertical strip cutting configuration
                vertical_strip_config = getattr(self.stage2_config, 'vertical_strip_cutting', {})
                if isinstance(vertical_strip_config, dict):
                    strip_padding = vertical_strip_config.get('padding', 20)
                    strip_min_width = vertical_strip_config.get('min_width', 1)
                    use_longest_lines_only = vertical_strip_config.get('use_longest_lines_only', False)
                    min_length_ratio = vertical_strip_config.get('min_length_ratio', 0.9)
                else:
                    strip_padding = 20
                    strip_min_width = 1
                    use_longest_lines_only = False
                    min_length_ratio = 0.9
                
                if self.stage2_config.enable_vertical_strip_cutting and recovered_result.get("cells"):
                    # Extract vertical lines from recovered table data
                    # This gives us more accurate boundaries considering merged cells
                    recovered_v_lines = get_major_vertical_boundaries(recovered_result)
                    
                    if self.stage2_config.verbose:
                        print(f"  Extracted {len(recovered_v_lines)} vertical lines from recovered table")
                    
                    strips_result = cut_vertical_strips(
                        image=refined_deskewed,
                        structure_json_path=None,  # Don't use structure_json, let it be created from v_lines
                        padding=strip_padding,
                        min_width=strip_min_width,
                        use_longest_lines_only=use_longest_lines_only,
                        min_length_ratio=min_length_ratio,
                        v_lines=recovered_v_lines,  # Use recovered lines
                        output_dir=self.stage2_config.output_dir / "05_vertical_strips",
                        base_name=base_name,
                        verbose=self.stage2_config.verbose
                    )
                    
                    if self.stage2_config.verbose:
                        print(f"  Vertical strips: {strips_result['num_strips']} columns saved")
                else:
                    if self.stage2_config.verbose:
                        print(f"  Vertical strip cutting skipped (disabled or no table structure)")
                
                # Step 6: Binarization of vertical strips (final step)
                if self.stage2_config.enable_binarization:
                    # Binarization only processes vertical strips
                    strips_dir = self.stage2_config.output_dir / "05_vertical_strips"
                    # Get all image files in the strips directory (both PNG and JPG)
                    strip_files = []
                    if strips_dir.exists():
                        strip_files = list(strips_dir.glob("*.png")) + list(strips_dir.glob("*.jpg"))
                    
                    if strip_files:
                        binarized_dir = self.stage2_config.output_dir / "06_binarized"
                        binarized_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Process each vertical strip
                        for strip_file in strip_files:
                            strip_img = load_image(strip_file)
                            
                            if self.stage2_config.save_debug_images:
                                # Use processor for debug mode
                                binarized_strip = self.binarize_processor_s2.process(
                                    strip_img,
                                    method=self.stage2_config.binarization_method,
                                    threshold=self.stage2_config.binarization_threshold,
                                    adaptive_block_size=self.stage2_config.binarization_adaptive_block_size,
                                    adaptive_c=self.stage2_config.binarization_adaptive_c,
                                    invert=self.stage2_config.binarization_invert,
                                    denoise=self.stage2_config.binarization_denoise,
                                )
                                
                                # Save debug images if debug directory is configured
                                if self.debug_run_dir_s2 and strip_file == strip_files[0]:  # Only save debug for first strip
                                    debug_subdir = self.debug_run_dir_s2 / "06_binarization" / base_name
                                    self.binarize_processor_s2.save_debug_images_to_dir(debug_subdir)
                            else:
                                # Use direct function call for normal processing
                                binarized_strip = binarize_image(
                                    strip_img,
                                    method=self.stage2_config.binarization_method,
                                    threshold=self.stage2_config.binarization_threshold,
                                    adaptive_block_size=self.stage2_config.binarization_adaptive_block_size,
                                    adaptive_c=self.stage2_config.binarization_adaptive_c,
                                    invert=self.stage2_config.binarization_invert,
                                    denoise=self.stage2_config.binarization_denoise,
                                )
                            
                            # Save binarized strip
                            binarized_strip_path = binarized_dir / strip_file.name
                            save_image(binarized_strip, binarized_strip_path)
                        
                        if self.stage2_config.verbose:
                            print(f"  Binarized {len(strip_files)} vertical strips using {self.stage2_config.binarization_method} method")
                            print(f"  Saved to: {binarized_dir.name}")
                    else:
                        if self.stage2_config.verbose:
                            print(f"  Binarization skipped (no vertical strips found)")
                else:
                    if self.stage2_config.verbose:
                        print(f"  Binarization skipped (disabled)")

            except Exception as e:
                print(f"Error refining {image_path}: {e}")

        if self.stage2_config.verbose:
            print(
                f"\n*** STAGE 2 COMPLETE: {len(refined_tables)} publication-ready tables ***"
            )
            print(f"Final output: {self.stage2_config.output_dir / '04_table_recovered'}")
            
            # Check if vertical strip cutting was enabled
            vertical_strip_config = getattr(self.stage2_config, 'vertical_strip_cutting', {})
            if isinstance(vertical_strip_config, dict) and vertical_strip_config.get('enable', True):
                print(f"Vertical strips: {self.stage2_config.output_dir / '05_vertical_strips'}")
            
            # Check if binarization was enabled
            if self.stage2_config.enable_binarization:
                print(f"Binarized output: {self.stage2_config.output_dir / '06_binarized'}")

        # Save run info at completion
        self._save_run_info("stage2", input_dir, len(image_files), "completed")

        return refined_tables

    def run_complete_pipeline(self, input_path: Path = None) -> List[Path]:
        """
        Run both Stage 1 and Stage 2 sequentially.

        Args:
            input_path: Input directory or single image file

        Returns:
            List of final refined table image paths
        """
        print("*** RUNNING COMPLETE TWO-STAGE PIPELINE ***")
        print("=" * 60)

        # Run Stage 1
        stage1_outputs = self.run_stage1(input_path)

        if not stage1_outputs:
            raise RuntimeError("Stage 1 produced no output. Cannot proceed to Stage 2.")

        print("\nStage 1 -> Stage 2 transition")
        print(f"   {len(stage1_outputs)} cropped tables ready for refinement")

        # Run Stage 2 - use the actual output directory from Stage 1
        stage1_output_dir = self.stage1_config.output_dir / "07_border_cropped"
        stage2_outputs = self.run_stage2(input_dir=stage1_output_dir)

        print("\n*** COMPLETE PIPELINE FINISHED! ***")
        print(f"   {len(stage2_outputs)} publication-ready tables generated")
        print(
            f"   Check final results in: {self.stage2_config.output_dir / '04_table_recovered'}"
        )

        return stage2_outputs


def main() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="OCR Table Extraction Pipeline")
    parser.add_argument(
        "input", nargs="?", default="input", help="Input directory or file"
    )
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Save debug images")

    args = parser.parse_args()

    # Create configuration
    config = Config(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        verbose=args.verbose,
        save_debug_images=args.debug,
    )

    # Run pipeline
    pipeline = OCRPipeline(config)

    input_path = Path(args.input)
    if input_path.is_file():
        # Process single file
        outputs = pipeline.process_image(input_path)
        print(f"Processed {input_path} -> {len(outputs)} output files")
    else:
        # Process directory
        outputs = pipeline.process_directory(input_path)
        print(f"Processed {len(outputs)} files total")


if __name__ == "__main__":
    main()
