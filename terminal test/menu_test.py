import shutil
from pathlib import Path

from src.storage.db_init import start_connection
from src.storage import database as db
from src.processing.menu_ingestion import extract_menu

# Cross-platform paths
DATA_DIR = Path("data") / "my_files"
STAGING_DIR = Path("data") / "staging_area"
PROCESSED_DIR = Path("data") / "processed_area"


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

    # checks if db exists, if not creates it
    temp_restaurant_id = start_connection(create_dummy=True)

    # 2nd loop: process files from staging
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in STAGING_DIR.iterdir():
        print(f"Processing {file_path}...")

        try:
            menu_df = extract_menu(temp_restaurant_id, file_path)
        except NotImplementedError as e:
            print(f"[SKIP] {e}")
            # Move to processed and continue
            shutil.move(str(file_path), PROCESSED_DIR / file_path.name)
            continue
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            shutil.move(str(file_path), PROCESSED_DIR / file_path.name)
            continue

        # Save using DataFrame-only signature; save_menu_db will use restaurant_id inside df
        res = db.save_menu_db(menu_df)
        print(f"Saved: {file_path} -> {res}")

        # Move file to processed
        shutil.move(str(file_path), PROCESSED_DIR / file_path.name)

    print("Done!")


if __name__ == "__main__":
    run_pipeline()