from pathlib import Path
from typing import List
import torch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from langchain_core.documents import Document


def _setup_mps_device() -> str:
    """Setup PyTorch MPS device if available (Apple Silicon).
    
    Returns:
        Device string: "mps" if available and working, "cpu" otherwise.
    """
    if torch.backends.mps.is_available():
        try:
            # Test MPS availability
            torch.zeros(1, device="mps")
            print("Using PyTorch MPS (Metal Performance Shaders) for acceleration")
            return "mps"
        except RuntimeError:
            print("MPS not working, falling back to CPU")
            return "cpu"
    return "cpu"


def _get_device() -> str:
    """Get the optimal device for the system."""
    device = _setup_mps_device()
    torch.set_default_device(device)
    return device

def _convert_pdf(pdf_path: Path, results_dir: str) -> List[Document]:
    """Convert a single PDF into its own subfolder, saving markdown + extracted images.

    Output layout:
        <results_dir>/<pdf-stem>/
            <pdf-stem>.md       ← markdown with relative image links
            image_000000_<hash>.png
            image_000001_<hash>.png
            ...

    Args:
        pdf_path: Path object pointing to the PDF file.
        results_dir: Root results directory; a per-PDF subfolder is created here.

    Returns:
        List with one LangChain Document (the markdown content).
    """
    # Initialize MPS device if available
    _get_device()
    
    pdf_out_dir = Path(results_dir) / pdf_path.stem
    pdf_out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_opts = PdfPipelineOptions(generate_picture_images=True)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    result = converter.convert(str(pdf_path))

    out_md = pdf_out_dir / f"{pdf_path.stem}.md"
    result.document.save_as_markdown(
        out_md,
        artifacts_dir=Path("images"),
        image_mode=ImageRefMode.REFERENCED,
    )
    print(f"Saved: {out_md}")

    md_text = out_md.read_text(encoding="utf-8")
    return [Document(page_content=md_text, metadata={"source": str(pdf_path)})]


def convert_directory(pdf_dir: str, results_dir: str) -> List[Document]:
    """Convert all PDFs in pdf_dir, saving each into its own results subfolder.

    Args:
        pdf_dir: Path to the directory containing PDF files.
        results_dir: Root directory where per-PDF result folders are created.

    Returns:
        List of LangChain Document objects for all converted PDFs.
    """
    pdf_dir_path = Path(pdf_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    all_pdf_files = sorted(pdf_dir_path.glob("*.pdf"))
    if not all_pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []

    all_docs: List[Document] = []
    for p in all_pdf_files:
        out_dir = results_path / p.stem
        if out_dir.exists():
            print(f"Skipping (already converted): {p.name}")
            continue
        all_docs.extend(_convert_pdf(p, results_dir))

    if not all_docs:
        print("All PDFs already converted, nothing to do.")

    return all_docs


def convert_single(pdf_path: str, results_dir: str) -> List[Document]:
    """Convert a single PDF into its own results subfolder.

    Args:
        pdf_path: Path to the PDF file.
        results_dir: Root directory where the per-PDF result folder is created.

    Returns:
        List of LangChain Document objects for the converted PDF.
    """
    return _convert_pdf(Path(pdf_path), results_dir)


if __name__ == "__main__":
    # Setup MPS device for conversion
    device = _get_device()
    print(f"Conversion will run on: {device}\n")
    
    # Example usage
    convert_single("Mem0_ AI_ML & Generative AI Developer Intern Assignment.pdf", "things/")
    # convert_directory("data", "results")