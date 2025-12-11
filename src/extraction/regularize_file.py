# regularize_file.py

import os
import shutil
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import DictionaryObject, NameObject
import re
import pdfplumber
from typing import Tuple, List, Optional

STAGING_DIR = Path("data\staging_area")
PROCESSED_DIR = Path("data\processed_area")


def detect_invoice_page_groups(p: str, reader: pdfplumber.PDF) -> Tuple[Tuple[int, ...], ...]:
    """
    Scans a PDF object page-by-page to group pages into distinct invoices based on 
    detected Invoice or Order IDs.

    This function implements a "State Change Detection" algorithm. It assumes adjacent 
    pages belong to the same invoice unless a *new* and *different* Invoice ID is detected.

    Args:
        p (str): The file path string. Not used for processing, but useful for 
                logging/error reporting (e.g., 'Processing file: {p}').
        reader (pdfplumber.PDF): The open pdfplumber PDF object to be scanned.

    Returns:
        Tuple[Tuple[int, ...], ...]: A tuple of tuples containing 1-based page indices.
        
        Example Return:
            ((1, 2), (3,), (4, 5, 6))
            - Pages 1 & 2 are Invoice A.
            - Page 3 is Invoice B.
            - Pages 4, 5, & 6 are Invoice C.
    """
    
    # ==============================================================================
    # 1. DEFINE REGEX PATTERNS
    # ==============================================================================
    # We use a list to prioritize patterns. The code stops at the first match found.
    # The '(?:...)' syntax groups tokens without capturing them.
    # The '(...)' syntax is the CAPTURE GROUP, which is the actual ID we extract.
    patterns = [
        # ------------------------------------------------------------------
        # PATTERN A: Standard Format
        # Matches: "Order No: 12345", "Invoice # 999", "Invoice: 555"
        # Breakdown:
        #   (?:Order|Invoice)   -> Look for word Order OR Invoice
        #   \s* -> Allow any amount of whitespace
        #   (?:No\.?|Number|#)? -> Optionally match "No.", "Number", or "#"
        #   \s*[:.]?\s* -> Allow whitespace, optional colon/dot separator
        #   (\d+)               -> CAPTURE group: The digits (The ID)
        # ------------------------------------------------------------------
        re.compile(r'(?:Order|Invoice)\s*(?:No\.?|Number|#)?\s*[:.]?\s*(\d+)', re.IGNORECASE),

        # ------------------------------------------------------------------
        # PATTERN B: Inverted Format
        # Matches: "1203379346 Order Number"
        # Breakdown:
        #   (\d{3,})            -> CAPTURE group: 3 or more digits. 
        #                          (We force 3+ digits to avoid matching "1 Order Number" as a quantity)
        #   \s+                 -> Must have whitespace
        #   (?:Order|Invoice)   -> Followed by Order or Invoice
        #   \s+Number           -> Followed by word Number
        # ------------------------------------------------------------------
        re.compile(r'(\d{3,})\s+(?:Order|Invoice)\s+Number', re.IGNORECASE),

        # ------------------------------------------------------------------
        # PATTERN C: The "Weird" Edge Case
        # Matches: "ORDERED perof1377184" or "Ship Via:# 1INVOICE 377184"
        # Requirement: Ignore the specific prefix "perof1" or "1" and capture the rest.
        # Breakdown:
        #   perof1              -> Literally match the noise string "perof1"
        #   (\d+)               -> CAPTURE group: The digits immediately following it.
        # ------------------------------------------------------------------
        re.compile(r'perof1(\d+)', re.IGNORECASE)
    ]

    # Initialize storage
    grouped_invoices: List[Tuple[int, ...]] = [] # Final list of invoice groups
    current_group: List[int] = []                # The group currently being built
    current_tracking_id: Optional[str] = None    # The ID of the current group
    
    # ==============================================================================
    # 2. ITERATE PAGES
    # ==============================================================================
    # We use enumerate(start=1) so 'i' represents the human-readable page number.
    for i, page in enumerate(reader.pages, start=1):
        
        # Extract text safely (handles blank pages or images without error)
        text = page.extract_text() or ""
        
        # Data Cleaning:
        # Replace newlines with spaces to handle headers that might wrap weirdly.
        # Strip creates a clean string for regex matching.
        text = text.replace('\n', ' ').strip()
        
        found_id = None
        
        # --- Extraction Logic ---
        # Try every pattern in the list until one works.
        for pat in patterns:
            match = pat.search(text)
            if match:
                # We extract group(1) which corresponds to the (...) in the regex.
                # .strip() removes any accidental whitespace caught in the capture group.
                found_id = match.group(1).strip()
                break # Stop searching patterns for this page

        # ==========================================================================
        # 3. DECISION LOGIC (State Machine)
        # ==========================================================================
        
        # CASE 1: Initialization
        # If this is the very first page of the PDF, start the first group.
        if i == 1:
            current_tracking_id = found_id
            current_group.append(i)
            continue
            
        # CASE 2: Continuation (No ID)
        # If the page has NO invoice number, we assume it is an overflow page 
        # (Page 2 of 2) belonging to the previous invoice.
        if found_id is None:
            current_group.append(i)
            
        # CASE 3: Continuation (Same ID)
        # If the page has an ID, and it matches the one we are currently tracking,
        # it is part of the same invoice (e.g., repeated header).
        elif found_id == current_tracking_id:
            current_group.append(i)
            
        # CASE 4: Split / New Invoice
        # We found an ID, and it is DIFFERENT from the current tracker.
        else:
            # A. Archive the previous group (convert list to tuple)
            if current_group:
                grouped_invoices.append(tuple(current_group))
            
            # B. Start a new group with the current page
            current_group = [i]
            
            # C. Update the tracker to the new ID
            current_tracking_id = found_id

    # ==============================================================================
    # 4. FINAL CLEANUP
    # ==============================================================================
    # The loop finishes after the last page, but the last group is still sitting 
    # in 'current_group'. We must add it to the final list.
    if current_group:
        grouped_invoices.append(tuple(current_group))

    # Convert the list of tuples into the requested tuple of tuples
    return tuple(grouped_invoices)


def split_pdf_by_page_groups(p, reader, groups):
    """
    Split a multi-page PDF into files according to `groups`.

    `groups` is a tuple of tuples of 1-based page indices (consecutive within each tuple).
    This function writes one output PDF per group. Filenames follow the pattern:
        {base_name}-page{start}[-{end}].pdf
    For single-page groups, the filename will be `{base_name}-page{n}.pdf`.
    For multi-page groups (e.g. pages 1 and 2), the filename will be
    `{base_name}-page1-2.pdf`.

    Behavior and protections match the original implementation:
    - Attempts to preserve the original /Info metadata dictionary.
    - Writes to a temporary `.tmp-{filename}` then atomically replaces the final file.
    - Collects created files and on exception cleans them up and keeps the original PDF.
    - On full success removes the original PDF (`p.unlink()`).

    Parameters:
    - p: Path object for the original PDF.
    - reader: PdfReader for the original PDF.
    - groups: tuple of tuples with 1-based page indices.
    """
    base_name = p.stem

    # fetch original /Info (may be an IndirectObject)
    orig_info_obj = reader.trailer.get("/Info")
    orig_info_resolved = None
    if orig_info_obj:
        try:
            orig_info_resolved = orig_info_obj.get_object()
        except Exception:
            orig_info_resolved = orig_info_obj

    # build a DictionaryObject copy that preserves PDF object types
    info_copy = None
    if orig_info_resolved:
        info_copy = DictionaryObject()
        for k, v in orig_info_resolved.items():
            key = NameObject(k) if not isinstance(k, NameObject) else k
            try:
                val = v.get_object()
            except Exception:
                val = v
            info_copy[key] = val

    created = []
    try:
        # Iterate over groups and produce one PDF per group
        for group in groups:
            # group is a tuple of 1-based consecutive page indices
            if not group:
                continue

            start = group[0]
            end = group[-1]
            if start == end:
                new_filename = f"{base_name}-page{start}.pdf"
            else:
                new_filename = f"{base_name}-page{start}-{end}.pdf"

            temp_path = PROCESSED_DIR / f".tmp-{new_filename}"
            final_path = PROCESSED_DIR / new_filename

            writer = PdfWriter()

            # add each page in the group (convert 1-based index to 0-based)
            for page_index in group:
                page = reader.pages[page_index - 1]
                writer.add_page(page)

            # inject original /Info dict unchanged
            if info_copy is not None:
                info_ref = writer._add_object(info_copy)
                writer._info = info_copy
                writer._root_object.update({NameObject("/Info"): info_ref})

            # write to temporary file then atomically replace
            with open(temp_path, "wb") as output_file:
                writer.write(output_file)

            os.replace(temp_path, final_path)
            created.append(final_path)

        # all groups created successfully, remove original
        p.unlink()

    except Exception as exc:
        # cleanup partial outputs and keep original
        for f in created:
            try:
                f.unlink()
            except Exception:
                pass
        print(f"Error splitting {p}: {exc}")


def process_multi_page_pdf(p, reader):
    """
    Top-level multi-page PDF processor.

    - Calls `detect_invoice_page_groups(p, reader)` to obtain grouping
      information (tuple of tuples of 1-based consecutive page indices).
    - If grouping information is returned, calls `split_pdf_by_page_groups`
      to perform the actual splitting/writing.
    - If `detect_invoice_page_groups` returns an empty tuple or None, no action
      is taken (assume the PDF has been handled elsewhere).

    Parameters:
    - p: pathlib.Path to the PDF file.
    - reader: PdfReader for the PDF.
    """
    # Determine groups of pages that constitute individual invoices.
    groups = detect_invoice_page_groups(p, reader)

    # If detector returns falsy (None or empty), assume another mechanism handled it.
    if not groups:
        # If False or empty, assume the PDF has already been handled on its own.
        return

    # Validate groups are a tuple of tuples of positive integers and within range.
    # This guards against infinite loops and accidental bad detector outputs.
    num_pages = len(reader.pages)
    validated_groups = []
    covered_pages = set()
    for grp in groups:
        if not isinstance(grp, (tuple, list)):
            raise ValueError("Groups must be an iterable of iterables of integers.")
        if not grp:
            continue
        # ensure consecutive and in range
        for i in range(len(grp) - 1):
            if grp[i+1] != grp[i] + 1:
                raise ValueError("Each inner group must contain consecutive page indices.")
        for page_index in grp:
            if not isinstance(page_index, int) or page_index < 1 or page_index > num_pages:
                raise ValueError("Page indices must be 1-based integers within page range.")
            if page_index in covered_pages:
                raise ValueError("Overlapping page indices detected in groups.")
            covered_pages.add(page_index)
        validated_groups.append(tuple(grp))

    # Ensure groups cover only contiguous sequences and do not overlap, but they
    # are allowed to omit pages if detector decides they shouldn't be split here.

    groups = tuple(validated_groups)

    # Finally, split according to groups.
    split_pdf_by_page_groups(p, reader, groups)




def process_files_to_processed_folder():

    for p in STAGING_DIR.iterdir():
        if p.is_dir():
            continue

        suffix = p.suffix.lower()

        if suffix in {".png", ".jpg", ".jpeg"}:
            shutil.move(str(p), str(PROCESSED_DIR / p.name))

        elif suffix == ".pdf":
            reader = PdfReader(str(p))
            num_pages = len(reader.pages)

            if num_pages <= 1:
                shutil.move(str(p), str(PROCESSED_DIR / p.name))

            else:
                process_multi_page_pdf(p, reader)

