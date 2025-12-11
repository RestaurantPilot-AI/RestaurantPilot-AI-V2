import shutil
from pathlib import Path

from src import process_files_to_processed_folder, start_connection, process_invoice, get_structured_data_from_text, save_inv_li_to_db


DATA_DIR = Path("data\\my_files")
STAGING_DIR = Path("data\\staging_area")
PROCESSED_DIR = Path("data\\processed_area")

def reset_staging():
    """Delete staging_area if it exists, then recreate empty."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


def copy_all_to_staging():
    """Copy all files in DATA_DIR (including all sub-folders) into staging_area."""
    for file_path in DATA_DIR.rglob("*"):
        if file_path.is_file():
            dest = STAGING_DIR / file_path.name

            # handle name collision
            if dest.exists():
                dest = STAGING_DIR / f"{file_path.stem}_dup{file_path.suffix}"

            shutil.copy2(file_path, dest)


def run_pipeline():
    # Reset staging before run
    reset_staging()

    # 1st loop: copy everything to staging
    copy_all_to_staging()

    # regularize files and saves to PROCESSED_DIR
    process_files_to_processed_folder()

    # checks if db exists, if not creates it
    temp_restaurant_id = start_connection(create_dummy=True)

    # 2nd loop: process files from staging
    for file_path in PROCESSED_DIR.iterdir():
        print(f"Processing {file_path}...")

        extracted_text, filename, text_length, page_count, extraction_timestamp = process_invoice(file_path)
        inv_df, li_df = get_structured_data_from_text(
                                            extracted_text=extracted_text, 
                                            filename=filename, 
                                            text_length=text_length, 
                                            page_count=page_count, 
                                            extraction_timestamp=extraction_timestamp,
                                            restaurant_id=temp_restaurant_id)
        save_inv_li_to_db(inv_df, li_df)

        print(f"Done! file: {file_path}")

    # Clean staging after run
    reset_staging()


if __name__ == "__main__":
    run_pipeline()
