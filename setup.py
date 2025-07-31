"""Setup script for OCR table extraction pipeline."""

from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Professional two-stage OCR table extraction pipeline"

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ocr-table-extraction",
    version="2.0.0",
    author="OCR Pipeline Team",
    description="Professional two-stage OCR table extraction pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ocr-pipeline=ocr_pipeline.pipeline:main",
            "ocr-stage1=scripts.run_stage1:main",
            "ocr-stage2=scripts.run_stage2:main", 
            "ocr-complete=scripts.run_complete:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.json", "configs/*.md"],
    },
)