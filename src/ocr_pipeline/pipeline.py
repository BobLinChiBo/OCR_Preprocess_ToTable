"""Main OCR pipeline."""

import argparse
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
            image, self.config.gutter_search_start, self.config.gutter_search_end
        )

        output_paths = []

        # Process each page
        for i, page in enumerate([left_page, right_page], 1):
            # Deskew
            deskewed, _ = utils.deskew_image(
                page,
                self.config.angle_range,
                self.config.angle_step,
                self.config.min_angle_correction,
            )

            # Margin removal (preprocessing step)
            processing_image = deskewed
            if self.config.enable_margin_removal:
                processing_image = utils.remove_margin_aggressive(
                    deskewed,
                    blur_kernel_size=self.config.blur_kernel_size,
                    black_threshold=self.config.black_threshold,
                    content_threshold=self.config.content_threshold,
                    morph_kernel_size=self.config.morph_kernel_size,
                    min_content_area_ratio=self.config.min_content_area_ratio,
                    padding=self.config.margin_padding,
                )

                if self.config.verbose:
                    print(f"    Margin removed: {processing_image.shape} (from {deskewed.shape})")

            # Detect table lines
            h_lines, v_lines = utils.detect_table_lines(
                processing_image,
                threshold=self.config.threshold,
                horizontal_kernel_size=self.config.horizontal_kernel_size,
                vertical_kernel_size=self.config.vertical_kernel_size,
                alignment_threshold=self.config.alignment_threshold,
                pre_merge_length_ratio=self.config.pre_merge_length_ratio,
                post_merge_length_ratio=self.config.post_merge_length_ratio,
                min_aspect_ratio=self.config.min_aspect_ratio,
            )

            # Crop to table region
            if h_lines and v_lines:
                cropped = utils.crop_table_region(processing_image, h_lines, v_lines)
            else:
                cropped = processing_image

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

        # Create separate pipeline instances for each stage
        self.stage1_pipeline = OCRPipeline(self.stage1_config)
        self.stage2_pipeline = OCRPipeline(self.stage2_config)

    def run_stage1(self, input_dir: Path = None) -> List[Path]:
        """
        Run Stage 1: Initial processing and table cropping.

        Args:
            input_dir: Input directory with raw images (optional)

        Returns:
            List of cropped table image paths ready for Stage 2
        """
        if self.stage1_config.verbose:
            print("*** STARTING STAGE 1: INITIAL PROCESSING ***")
            print("=" * 50)

        # Process images through complete Stage 1 pipeline
        input_dir = input_dir or self.stage1_config.input_dir
        cropped_tables = []

        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        image_files = utils.get_image_files(input_dir)
        if not image_files:
            raise ValueError(f"No image files found in: {input_dir}")

        if self.stage1_config.verbose:
            print(f"Found {len(image_files)} images to process")

        # Process each image through Stage 1 steps
        for image_path in image_files:
            try:
                if self.stage1_config.verbose:
                    print(f"\nProcessing: {image_path.name}")

                # Load and split image
                image = utils.load_image(image_path)
                left_page, right_page = utils.split_two_page_image(
                    image,
                    self.stage1_config.gutter_search_start,
                    self.stage1_config.gutter_search_end,
                )

                # Save split pages
                split_dir = self.stage1_config.output_dir / "01_split_pages"
                for i, page in enumerate([left_page, right_page], 1):
                    split_path = split_dir / f"{image_path.stem}_page_{i}.jpg"
                    utils.save_image(page, split_path)
                    if self.stage1_config.verbose:
                        print(f"  Split page saved: {split_path.name}")

                # Process each split page
                for i, page in enumerate([left_page, right_page], 1):
                    page_name = f"{image_path.stem}_page_{i}"

                    # Deskew
                    deskewed, _ = utils.deskew_image(
                        page,
                        self.stage1_config.angle_range,
                        self.stage1_config.angle_step,
                        self.stage1_config.min_angle_correction,
                    )

                    # Save deskewed image
                    deskew_dir = self.stage1_config.output_dir / "02_deskewed"
                    deskew_path = deskew_dir / f"{page_name}_deskewed.jpg"
                    utils.save_image(deskewed, deskew_path)

                    # Margin removal
                    processing_image = deskewed
                    if self.stage1_config.enable_margin_removal:
                        processing_image = utils.remove_margin_aggressive(
                            deskewed,
                            blur_kernel_size=self.stage1_config.blur_kernel_size,
                            black_threshold=self.stage1_config.black_threshold,
                            content_threshold=self.stage1_config.content_threshold,
                            morph_kernel_size=self.stage1_config.morph_kernel_size,
                            min_content_area_ratio=self.stage1_config.min_content_area_ratio,
                            padding=self.stage1_config.margin_padding,
                        )

                        # Save margin-removed image
                        margin_dir = self.stage1_config.output_dir / "03_margin_removed"
                        margin_path = margin_dir / f"{page_name}_margin_removed.jpg"
                        utils.save_image(processing_image, margin_path)

                        if self.stage1_config.verbose:
                            print(f"    Margin removed: {processing_image.shape} (from {deskewed.shape})")

                    # Table line detection
                    h_lines, v_lines = utils.detect_table_lines(
                        processing_image,
                        threshold=self.stage1_config.threshold,
                        horizontal_kernel_size=self.stage1_config.horizontal_kernel_size,
                        vertical_kernel_size=self.stage1_config.vertical_kernel_size,
                        alignment_threshold=self.stage1_config.alignment_threshold,
                        pre_merge_length_ratio=self.stage1_config.pre_merge_length_ratio,
                        post_merge_length_ratio=self.stage1_config.post_merge_length_ratio,
                        min_aspect_ratio=self.stage1_config.min_aspect_ratio,
                    )

                    # Save table line visualization
                    lines_dir = self.stage1_config.output_dir / "04_table_lines"
                    lines_path = lines_dir / f"{page_name}_table_lines.jpg"
                    vis_image = utils.visualize_detected_lines(processing_image, h_lines, v_lines)
                    utils.save_image(vis_image, lines_path)

                    # Final table cropping
                    if h_lines and v_lines:
                        cropped_table = utils.crop_table_region(
                            processing_image, h_lines, v_lines
                        )
                    else:
                        cropped_table = processing_image

                    # Save final cropped table for Stage 2
                    crop_dir = self.stage1_config.output_dir / "05_cropped_tables"
                    crop_path = crop_dir / f"{page_name}_cropped.jpg"
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
            print(f"Output: {self.stage1_config.output_dir / '05_cropped_tables'}")

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

                # Optional additional margin removal for Stage 2
                processing_image = refined_deskewed
                if self.stage2_config.enable_margin_removal:
                    processing_image = utils.remove_margin_aggressive(
                        refined_deskewed,
                        blur_kernel_size=self.stage2_config.blur_kernel_size,
                        black_threshold=self.stage2_config.black_threshold,
                        content_threshold=self.stage2_config.content_threshold,
                        morph_kernel_size=self.stage2_config.morph_kernel_size,
                        min_content_area_ratio=self.stage2_config.min_content_area_ratio,
                        padding=self.stage2_config.margin_padding,
                    )

                    # Save margin-removed image
                    margin_dir = self.stage2_config.output_dir / "02_margin_removed"
                    margin_path = margin_dir / f"{base_name}_margin_removed.jpg"
                    utils.save_image(processing_image, margin_path)

                    if self.stage2_config.verbose:
                        print(f"    Stage 2 margin removed: {processing_image.shape} (from {refined_deskewed.shape})")

                # Refined line detection with tighter parameters
                h_lines, v_lines = utils.detect_table_lines(
                    processing_image,
                    threshold=self.stage2_config.threshold,
                    horizontal_kernel_size=self.stage2_config.horizontal_kernel_size,
                    vertical_kernel_size=self.stage2_config.vertical_kernel_size,
                    alignment_threshold=self.stage2_config.alignment_threshold,
                    pre_merge_length_ratio=self.stage2_config.pre_merge_length_ratio,
                    post_merge_length_ratio=self.stage2_config.post_merge_length_ratio,
                    min_aspect_ratio=self.stage2_config.min_aspect_ratio,
                )

                # Save refined line detection visualization
                lines_dir = self.stage2_config.output_dir / "03_table_lines"
                lines_path = lines_dir / f"{base_name}_refined_table_lines.jpg"
                vis_image = utils.visualize_detected_lines(processing_image, h_lines, v_lines)
                utils.save_image(vis_image, lines_path)

                # Table fitting for publication-ready output
                fitting_dir = self.stage2_config.output_dir / "04_fitted_tables"
                fitted_path = fitting_dir / f"{base_name}_fitted.jpg"

                # Use the processed image (after margin removal if enabled) as the final result
                utils.save_image(processing_image, fitted_path)
                refined_tables.append(fitted_path)

                if self.stage2_config.verbose:
                    print(f"  Refined table: {fitted_path.name}")

            except Exception as e:
                print(f"Error refining {image_path}: {e}")

        if self.stage2_config.verbose:
            print(
                f"\n*** STAGE 2 COMPLETE: {len(refined_tables)} publication-ready tables ***"
            )
            print(f"Final output: {self.stage2_config.output_dir / '04_fitted_tables'}")

        return refined_tables

    def run_complete_pipeline(self, input_dir: Path = None) -> List[Path]:
        """
        Run both Stage 1 and Stage 2 sequentially.

        Args:
            input_dir: Input directory with raw images

        Returns:
            List of final refined table image paths
        """
        print("*** RUNNING COMPLETE TWO-STAGE PIPELINE ***")
        print("=" * 60)

        # Run Stage 1
        stage1_outputs = self.run_stage1(input_dir)

        if not stage1_outputs:
            raise RuntimeError("Stage 1 produced no output. Cannot proceed to Stage 2.")

        print("\nStage 1 -> Stage 2 transition")
        print(f"   {len(stage1_outputs)} cropped tables ready for refinement")

        # Run Stage 2
        stage2_outputs = self.run_stage2()

        print("\n*** COMPLETE PIPELINE FINISHED! ***")
        print(f"   {len(stage2_outputs)} publication-ready tables generated")
        print(
            f"   Check final results in: {self.stage2_config.output_dir / '04_fitted_tables'}"
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
