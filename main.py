import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile

from utils import log, setup_logger
from processing import process_zip_file
from config import TEMP_DIR, SUPPORTED_FILE_EXTENSIONS, OUTPUT_FILENAME # Import OUTPUT_FILENAME

# Ensure temp processing directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize logger
setup_logger()

app = FastAPI(title="Document Processing Service", version="2.0.0")

def cleanup_file(file_path: str):
    """Background task to delete a file."""
    try:
        if os.path.exists(file_path): # Check if file exists before removing
            os.remove(file_path)
            log.info(f"Cleaned up temporary file: {file_path}")
        else:
            log.info(f"Cleanup skipped, file already removed: {file_path}")
    except OSError as e:
        log.error(f"Error cleaning up file {file_path}: {e}")

@app.post("/process-zip/", response_class=FileResponse)
async def create_upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a ZIP file containing case folders with document files (PDF, PNG, JPEG).
    Processes them using Vertex AI (Classification then Extraction).
    Returns a CSV spreadsheet with extracted data, confidence, and reasoning.
    """
    if not file.filename.endswith(".zip"):
        log.error(f"Invalid file type uploaded: {file.filename}. Only .zip files are accepted.")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a ZIP file.")

    log.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")

    temp_zip_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=TEMP_DIR) as temp_zip_file:
            shutil.copyfileobj(file.file, temp_zip_file)
            temp_zip_path = temp_zip_file.name
        log.info(f"Saved uploaded zip file temporarily to: {temp_zip_path}")

    except Exception as e:
         log.exception(f"Failed to save uploaded file {file.filename}: {e}")
         # Ensure cleanup if temp file was partially created but saving failed
         if temp_zip_path and os.path.exists(temp_zip_path):
             background_tasks.add_task(cleanup_file, temp_zip_path)
         raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        await file.close()

    if not temp_zip_path:
         raise HTTPException(status_code=500, detail="Failed to create temporary file path.")

    try:
        log.info(f"Starting processing for temporary zip: {temp_zip_path}")
        output_csv_path = process_zip_file(temp_zip_path)
        log.info(f"Processing complete. Output CSV at: {output_csv_path}")

        background_tasks.add_task(cleanup_file, output_csv_path)
        background_tasks.add_task(cleanup_file, temp_zip_path)

        # Determine media type dynamically based on OUTPUT_FILENAME extension
        media_type = 'text/csv' if OUTPUT_FILENAME.endswith('.csv') else 'application/octet-stream' # Default if not CSV
        
        return FileResponse(
            path=output_csv_path,
            filename=os.path.basename(output_csv_path),
            media_type=media_type
        )

    except ValueError as ve:
         log.error(f"Value Error during processing: {ve}")
         background_tasks.add_task(cleanup_file, temp_zip_path)
         raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
         log.error(f"Runtime Error during processing: {re}")
         background_tasks.add_task(cleanup_file, temp_zip_path)
         raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        log.exception(f"An unexpected error occurred during processing zip file {temp_zip_path}: {e}")
        background_tasks.add_task(cleanup_file, temp_zip_path)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Document Processing API (v2.1 - Multiformat Document Extraction)",
        "supported_formats": f"PDF, PNG, JPEG files are supported for processing",
        "endpoint": "Use the /process-zip/ endpoint to upload your ZIP file containing documents"
    }

# --- To run the server ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8000