"""Main OCR pipeline."""

import argparse
import json
from pathlib import Path
from typing import List

from .config import (
    Config,
    get_default_config,
    Stage1Config,
    Stage2Config,
    get_stage1_config,
    get_stage2_config,
)
from . import utils


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
        image = utils.load_image(image_path)

        # Split into two pages
        left_page, right_page = utils.split_two_page_image(
            image,
            search_ratio=self.config.search_ratio,
            blur_k=self.config.blur_k,
            open_k=self.config.open_k,
            width_min=self.config.width_min
        )

        output_paths = []

        # Process each page
        for i, page in enumerate([left_page, right_page], 1):
            # Margin removal (preprocessing step)
            processing_image = page
            if self.config.enable_margin_removal:
                processing_image = utils.remove_margin_inscribed(
                    page,
                    blur_ksize=getattr(self.config, 'inscribed_blur_ksize', 5),
                    close_ksize=getattr(self.config, 'inscribed_close_ksize', 25),
                    close_iter=getattr(self.config, 'inscribed_close_iter', 2),
                )

                if self.config.verbose:
                    print(f"    Margin removed: {processing_image.shape} (from {page.shape})")

            # Deskew
            deskewed, _ = utils.deskew_image(
                processing_image,
                self.config.angle_range,
                self.config.angle_step,
                self.config.min_angle_correction,
            )

            # Detect table lines
            h_lines, v_lines = utils.detect_table_lines(
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
            )

            # Use deskewed image directly (table cropping is done in Stage 1)
            cropped = deskewed

            # Save result
            output_name = f"{image_path.stem}_page_{i}.jpg"
            output_path = self.config.output_dir / output_name
            utils.save_image(cropped, output_path)
            output_paths.append(output_path)

            if self.config.verbose:
                print(f"  Saved: {output_path}")

        return output_paths

    def process_directory(self, input_dir: Path = None) -> List[Path]:
        """Process all images in input directory."""
        input_dir = input_dir or self.config.input_dir

        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        image_files = utils.get_image_files(input_dir)

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
            image_files = utils.get_image_files(input_path)
            if not image_files:
                raise ValueError(f"No image files found in: {input_path}")

        if self.stage1_config.verbose:
            print(f"Found {len(image_files)} images to process")

        # Process each image through Stage 1 steps
        for image_path in image_files:
            try:
                if self.stage1_config.verbose:
                    print(f"\nProcessing: {image_path.name}")

                # Load image
                image = utils.load_image(image_path)
                processing_image = image

                # Step 1: Mark removal (on full image)
                if self.stage1_config.enable_mark_removal:
                    processing_image = utils.remove_marks(
                        processing_image,
                        dilate_iter=self.stage1_config.mark_removal_dilate_iter
                    )
                    
                    # Save marks-removed image
                    marks_dir = self.stage1_config.output_dir / "01_marks_removed"
                    marks_path = marks_dir / f"{image_path.stem}_marks_removed.jpg"
                    utils.save_image(processing_image, marks_path)
                    
                    if self.stage1_config.verbose:
                        print(f"  Marks removed from full image")

                # Step 2: Margin removal (on cleaned full image)
                if self.stage1_config.enable_margin_removal:
                    processing_image = utils.remove_margin_inscribed(
                        processing_image,
                        blur_ksize=getattr(self.stage1_config, 'inscribed_blur_ksize', 7),
                        close_ksize=getattr(self.stage1_config, 'inscribed_close_ksize', 30),
                        close_iter=getattr(self.stage1_config, 'inscribed_close_iter', 3),
                    )
                    
                    # Save margin-removed image
                    margin_dir = self.stage1_config.output_dir / "02_margin_removed"
                    margin_path = margin_dir / f"{image_path.stem}_margin_removed.jpg"
                    utils.save_image(processing_image, margin_path)
                    
                    if self.stage1_config.verbose:
                        print(f"  Margin removed from full image: {processing_image.shape}")

                # Step 3: Split processed image into pages
                left_page, right_page = utils.split_two_page_image(
                    processing_image,
                    search_ratio=self.stage1_config.search_ratio,
                    blur_k=self.stage1_config.blur_k,
                    open_k=self.stage1_config.open_k,
                    width_min=self.stage1_config.width_min,
                )

                # Save split pages
                split_dir = self.stage1_config.output_dir / "03_split_pages"
                for i, page in enumerate([left_page, right_page], 1):
                    split_path = split_dir / f"{image_path.stem}_page_{i}.jpg"
                    utils.save_image(page, split_path)
                    if self.stage1_config.verbose:
                        print(f"  Split page saved: {split_path.name}")

                # Process each split page
                for i, page in enumerate([left_page, right_page], 1):
                    page_name = f"{image_path.stem}_page_{i}"
                    processing_image = page

                    # Step 4: Deskew (per page)
                    deskewed, angle = utils.deskew_image(
                        processing_image,
                        self.stage1_config.angle_range,
                        self.stage1_config.angle_step,
                        self.stage1_config.min_angle_correction,
                    )

                    # Save deskewed image
                    deskew_dir = self.stage1_config.output_dir / "04_deskewed"
                    deskew_path = deskew_dir / f"{page_name}_deskewed.jpg"
                    utils.save_image(deskewed, deskew_path)

                    if self.stage1_config.verbose:
                        print(f"  Deskewed: {angle:.2f} degrees")

                    # Step 5: Table line detection
                    h_lines, v_lines = utils.detect_table_lines(
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
                    )

                    # Save table line visualization
                    lines_dir = self.stage1_config.output_dir / "05_table_lines"
                    lines_path = lines_dir / f"{page_name}_table_lines.jpg"
                    vis_image = utils.visualize_detected_lines(deskewed, h_lines, v_lines)
                    utils.save_image(vis_image, lines_path)

                    if self.stage1_config.verbose:
                        print(f"  Table lines: {len(h_lines)} horizontal, {len(v_lines)} vertical")

                    # Save lines data to JSON for next step
                    lines_data_dir = lines_dir / "lines_data"
                    lines_data_dir.mkdir(parents=True, exist_ok=True)
                    lines_json_path = lines_data_dir / f"{page_name}_lines_data.json"
                    lines_data = {
                        "horizontal_lines": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in h_lines],
                        "vertical_lines": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in v_lines]
                    }
                    with open(lines_json_path, 'w') as f:
                        json.dump(lines_data, f, indent=2)

                    # Step 6: Table structure detection from detected lines
                    table_structure = utils.detect_table_structure(
                        h_lines,  # Pass horizontal lines
                        v_lines,  # Pass vertical lines
                        eps=getattr(self.stage1_config, 'table_detection_eps', 10),
                        return_analysis=True
                    )

                    # Save table structure visualization
                    structure_dir = self.stage1_config.output_dir / "06_table_structure"
                    structure_path = structure_dir / f"{page_name}_table_structure.jpg"
                    structure_vis = utils.visualize_table_structure(deskewed, table_structure)
                    utils.save_image(structure_vis, structure_path)

                    if self.stage1_config.verbose:
                        xs = table_structure.get("xs", [])
                        ys = table_structure.get("ys", [])
                        cells = table_structure.get("cells", [])
                        print(f"  Table structure: {len(cells)} cells in {len(xs)}x{len(ys)} grid")

                    # Save table structure data to JSON for next step
                    structure_data_dir = structure_dir / "structure_data"
                    structure_data_dir.mkdir(parents=True, exist_ok=True)
                    structure_json_path = structure_data_dir / f"{page_name}_structure_data.json"
                    with open(structure_json_path, 'w') as f:
                        json.dump(table_structure, f, indent=2)

                    # Step 7: Crop deskewed image using detected table borders with padding
                    cropped_table = utils.crop_to_table_borders(
                        deskewed, 
                        table_structure,
                        padding=getattr(self.stage1_config, 'table_crop_padding', 20),
                        return_analysis=False
                    )

                    # Save border-cropped table for Stage 2
                    crop_dir = self.stage1_config.output_dir / "07_border_cropped"
                    crop_path = crop_dir / f"{page_name}_border_cropped.jpg"
                    utils.save_image(cropped_table, crop_path)
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

        image_files = utils.get_image_files(input_dir)
        if not image_files:
            raise ValueError(f"No cropped table images found in: {input_dir}")

        if self.stage2_config.verbose:
            print(f"Found {len(image_files)} cropped tables from Stage 1")

        refined_tables = []

        # Process each cropped table through Stage 2 refinement
        for image_path in image_files:
            try:
                if self.stage2_config.verbose:
                    print(f"\nRefining: {image_path.name}")

                # Load cropped table
                table_image = utils.load_image(image_path)
                base_name = image_path.stem.replace("_cropped", "")

                # Re-deskew for fine-tuning
                refined_deskewed, _ = utils.deskew_image(
                    table_image,
                    self.stage2_config.angle_range,
                    self.stage2_config.angle_step,
                    self.stage2_config.min_angle_correction,
                )

                # Save refined deskewed
                deskew_dir = self.stage2_config.output_dir / "01_deskewed"
                deskew_path = deskew_dir / f"{base_name}_refined_deskewed.jpg"
                utils.save_image(refined_deskewed, deskew_path)

                # Refined line detection with tighter parameters
                h_lines, v_lines = utils.detect_table_lines(
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
                )

                # Save refined line detection visualization
                lines_dir = self.stage2_config.output_dir / "02_table_lines"
                lines_path = lines_dir / f"{base_name}_refined_table_lines.jpg"
                vis_image = utils.visualize_detected_lines(refined_deskewed, h_lines, v_lines)
                utils.save_image(vis_image, lines_path)

                # Table fitting for publication-ready output
                fitting_dir = self.stage2_config.output_dir / "03_fitted_tables"
                fitted_path = fitting_dir / f"{base_name}_fitted.jpg"

                # Use the refined deskewed image as the final result
                utils.save_image(refined_deskewed, fitted_path)
                refined_tables.append(fitted_path)

                if self.stage2_config.verbose:
                    print(f"  Refined table: {fitted_path.name}")

            except Exception as e:
                print(f"Error refining {image_path}: {e}")

        if self.stage2_config.verbose:
            print(
                f"\n*** STAGE 2 COMPLETE: {len(refined_tables)} publication-ready tables ***"
            )
            print(f"Final output: {self.stage2_config.output_dir / '03_fitted_tables'}")

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

        # Run Stage 2
        stage2_outputs = self.run_stage2()

        print("\n*** COMPLETE PIPELINE FINISHED! ***")
        print(f"   {len(stage2_outputs)} publication-ready tables generated")
        print(
            f"   Check final results in: {self.stage2_config.output_dir / '03_fitted_tables'}"
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
