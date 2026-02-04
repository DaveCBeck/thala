#!/usr/bin/env python3
"""Test smart routing with a varied batch of PDFs.

This script processes PDFs of various sizes and complexities to demonstrate
the CPU/GPU routing system.
"""

import asyncio
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main():
    # Import after logging setup
    from core.scraping.pdf import analyze_document, process_document_smart
    from core.scraping.pdf.analysis import DocumentComplexity

    # Define test PDFs - varied sizes and types
    downloads_dir = Path("/home/dave/thala/services/retrieve-academic/downloads")

    # CPU stress test: 50 copies of the same LIGHT PDF (routes to CPU)
    # This tests CPU parallelism with 16 workers
    test_pdfs = [
        ("10.1098_rsos.150664", "4892c39a2cf8100d33e3bebf6d057058.pdf"),  # 120KB, 18 pages, LIGHT
    ] * 50  # 50 copies to stress-test CPU parallelism

    print("\n" + "=" * 80)
    print("SMART ROUTING TEST - Processing varied PDF batch (PARALLEL)")
    print("=" * 80)

    # Load all PDFs and prepare tasks
    tasks = []
    pdf_info = []

    for doi_dir, filename in test_pdfs:
        pdf_path = downloads_dir / doi_dir / filename

        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue

        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        pdf_content = pdf_path.read_bytes()

        print(f"📄 Queuing: {doi_dir} ({size_mb:.2f} MB)")
        pdf_info.append({"doi": doi_dir, "size_mb": size_mb, "content": pdf_content})
        tasks.append(process_document_smart(pdf_content))

    print(f"\n🚀 Submitting {len(tasks)} PDFs for parallel processing...")
    total_start = time.perf_counter()

    # Process all concurrently
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.perf_counter() - total_start

    # Collect results
    results = []
    for info, result in zip(pdf_info, results_raw):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {info['doi']}: {result}")
            results.append({
                "doi": info["doi"],
                "size_mb": info["size_mb"],
                "error": str(result),
            })
        else:
            # Get analysis info from result
            analysis = result.analysis
            results.append({
                "doi": info["doi"],
                "size_mb": info["size_mb"],
                "complexity": analysis.complexity.value if analysis else "unknown",
                "pages": result.page_count,
                "path": result.processing_path,
                "chars": len(result.markdown),
            })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    cpu_count = sum(1 for r in results if r.get("path") == "cpu")
    gpu_count = sum(1 for r in results if r.get("path") == "gpu")
    fallback_count = sum(1 for r in results if r.get("path") == "gpu_fallback")
    error_count = sum(1 for r in results if "error" in r)

    print(f"\nRouting breakdown:")
    print(f"  - CPU fast-path: {cpu_count}")
    print(f"  - GPU (Marker):  {gpu_count}")
    print(f"  - GPU fallback:  {fallback_count}")
    print(f"  - Errors:        {error_count}")

    # Complexity breakdown
    complexity_counts = {}
    for r in results:
        if "complexity" in r:
            c = r["complexity"]
            complexity_counts[c] = complexity_counts.get(c, 0) + 1

    print(f"\nComplexity breakdown:")
    for c in ["light", "mixed", "heavy"]:
        print(f"  - {c.capitalize()}: {complexity_counts.get(c, 0)}")

    # Total chars processed
    total_chars = sum(r.get("chars", 0) for r in results)
    total_pages = sum(r.get("pages", 0) for r in results)

    print(f"\nOutput:")
    print(f"  - Total pages: {total_pages:,}")
    print(f"  - Total chars: {total_chars:,}")

    print(f"\n⏱️  Total wall-clock time: {total_time:.2f}s")
    if total_pages > 0:
        print(f"   Throughput: {total_pages / total_time:.1f} pages/sec")

    # Detailed results table
    print("\n" + "-" * 80)
    print(f"{'DOI':<45} {'Size':>8} {'Pages':>6} {'Cmplx':>8} {'Path':>12} {'Chars':>10}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['doi'][:44]:<45} {r['size_mb']:>7.2f}M {'-':>6} {'ERROR':>8} {'-':>12} {'-':>10}")
        else:
            print(f"{r['doi'][:44]:<45} {r['size_mb']:>7.2f}M {r['pages']:>6} {r['complexity']:>8} {r['path']:>12} {r['chars']:>10,}")


if __name__ == "__main__":
    asyncio.run(main())
