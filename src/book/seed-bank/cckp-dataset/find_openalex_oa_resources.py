"""
Finds OpenAlex open access resources for publications in the CCKP dataset.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyalex import Works
from pyalex.api import config as pyalex_config


def main() -> None:
    # Fixed paths and settings so this runs without flags.
    input_csv = "src/book/seed-bank/cckp-dataset/cckp_publications.csv"
    output_parquet = (
        "src/book/seed-bank/cckp-dataset/cckp_publications_openalex_oa.parquet"
    )
    # Batch output files live here.
    checkpoint_dir = "src/book/seed-bank/cckp-dataset/data"
    doi_column = "doi"
    sleep_s = 0.1
    email = os.getenv("OPENALEX_EMAIL")

    # OpenAlex requests require an email in the polite pool.
    if not email:
        raise SystemExit(
            "Missing OpenAlex email. Set --email or OPENALEX_EMAIL for polite access."
        )
    pyalex_config.email = email

    # Load the source data and verify the DOI column exists.
    df = pd.read_csv(input_csv)
    if doi_column not in df.columns:
        raise SystemExit(
            f"Missing DOI column '{doi_column}'. Available: {list(df.columns)}"
        )

    # Normalize DOIs to reduce duplicates and remove empty values.
    os.makedirs(checkpoint_dir, exist_ok=True)
    doi_prefix_re = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)
    df["doi_normalized"] = (
        df[doi_column]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})
        .str.replace(doi_prefix_re, "", regex=True)
        .str.lower()
    )
    df.loc[
        df["doi_normalized"].isna() | (df["doi_normalized"] == ""), "doi_normalized"
    ] = pd.NA
    dois = (
        df["doi_normalized"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    # Parallel fetch in batches, with checkpointed outputs.
    rows: List[Dict[str, Any]] = []
    batch_size = 200
    max_workers = 8

    def fetch_one(doi: str) -> Dict[str, Any]:
        # Look up a single DOI in OpenAlex and return the fields we care about.
        try:
            results = Works().filter(doi=doi).get(per_page=1)
        except Exception as exc:
            # Keep going even if OpenAlex returns a bad response.
            if sleep_s:
                time.sleep(max(sleep_s, 1.0))
            return {
                "doi_normalized": doi,
                "openalex_id": None,
                "is_oa": None,
                "oa_status": None,
                "oa_url": None,
                "pdf_url": None,
                "landing_page_url": None,
                "host_type": None,
                "license": None,
                "version": None,
                "openalex_error": str(exc),
            }

        if sleep_s:
            time.sleep(sleep_s)

        work = results[0] if results else None
        if not work:
            # No match for this DOI.
            return {
                "doi_normalized": doi,
                "openalex_id": None,
                "is_oa": None,
                "oa_status": None,
                "oa_url": None,
                "pdf_url": None,
                "landing_page_url": None,
                "host_type": None,
                "license": None,
                "version": None,
                "openalex_error": None,
            }

        # Prefer best OA location, then primary, then any location.
        open_access = work.get("open_access") or {}
        primary = work.get("primary_location") or {}
        best_oa = work.get("best_oa_location") or {}
        locations = work.get("locations") or []
        pdf_url = best_oa.get("pdf_url") or primary.get("pdf_url")
        landing_page_url = best_oa.get("landing_page_url") or primary.get(
            "landing_page_url"
        )
        if not pdf_url:
            pdf_url = next(
                (loc.get("pdf_url") for loc in locations if loc.get("pdf_url")), None
            )
        if not landing_page_url:
            landing_page_url = next(
                (
                    loc.get("landing_page_url")
                    for loc in locations
                    if loc.get("landing_page_url")
                ),
                None,
            )
        # If OA URL points directly to a PDF, use it.
        oa_url = open_access.get("oa_url")
        if not pdf_url and oa_url and oa_url.lower().endswith(".pdf"):
            pdf_url = oa_url

        # Return the row we will merge back into the dataset.
        return {
            "doi_normalized": doi,
            "openalex_id": work.get("id"),
            "is_oa": open_access.get("is_oa"),
            "oa_status": open_access.get("oa_status"),
            "oa_url": oa_url,
            "pdf_url": pdf_url,
            "landing_page_url": landing_page_url,
            "host_type": primary.get("host_type"),
            "license": primary.get("license"),
            "version": primary.get("version"),
            "openalex_error": None,
        }

    total = len(dois)
    for batch_start in range(0, total, batch_size):
        batch = dois[batch_start : batch_start + batch_size]
        # Launch a batch in parallel and stream progress.
        print(
            f"Starting batch {batch_start // batch_size + 1} "
            f"({batch_start + 1}-{batch_start + len(batch)} of {total})"
        )
        batch_rows: List[Tuple[int, Dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_one, doi): (batch_start + i + 1, doi)
                for i, doi in enumerate(batch)
            }
            for future in as_completed(futures):
                idx, doi = futures[future]
                row = future.result()
                batch_rows.append((idx, row))
                print(f"Completed {idx}/{total}: {doi}")

        batch_rows.sort(key=lambda item: item[0])
        rows.extend([row for _, row in batch_rows])

        # Write a batch checkpoint file for safety.
        oa_df = pd.DataFrame([row for _, row in batch_rows])
        merged = df[df["doi_normalized"].isin(batch)].merge(
            oa_df, on="doi_normalized", how="left"
        )
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"cckp_publications_openalex_oa_{batch_start + len(batch)}.parquet",
        )
        merged.to_parquet(checkpoint_path, index=False)
        print(f"Checkpoint: wrote {len(merged)} rows to {checkpoint_path}")

    # Final merged dataset for the full input.
    oa_df = pd.DataFrame(rows)
    merged = df.merge(oa_df, on="doi_normalized", how="left")
    merged.to_parquet(output_parquet, index=False)
    print(f"Wrote {len(merged)} rows to {output_parquet}")

    # Stream-append checkpoint files into a single aggregated parquet.
    aggregated_parquet = os.path.join(
        checkpoint_dir, "cckp_publications_openalex_oa_aggregated.parquet"
    )
    checkpoint_files = sorted(
        path
        for path in os.listdir(checkpoint_dir)
        if path.endswith(".parquet")
        and path.startswith("cckp_publications_openalex_oa_")
        and path != os.path.basename(aggregated_parquet)
    )
    if checkpoint_files:
        # ParquetWriter avoids loading all parts into memory at once.
        writer = None
        total_rows = 0
        for filename in checkpoint_files:
            path = os.path.join(checkpoint_dir, filename)
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(aggregated_parquet, table.schema)
            writer.write_table(table)
            total_rows += table.num_rows
        if writer is not None:
            writer.close()
        print(
            f"Aggregated {total_rows} rows from {len(checkpoint_files)} files to "
            f"{aggregated_parquet}"
        )


if __name__ == "__main__":
    main()
