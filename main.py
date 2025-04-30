# main.py
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile

from utils import log, setup_logger
from processing import process_zip_file # This now uses the new workflow
from config import TEMP_DIR

# Ensure temp processing directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize logger
setup_logger()

app = FastAPI(title="Document Processing Service", version="2.0.0") # Version bump might be nice

def cleanup_file(file_path: str):
    """Background task to delete a file."""
    try:
        if os.path.exists(file_path): # Check if file exists before removing
            os.remove(file_path)
            log.info(f"Cleaned up temporary file: {file_path}")
        # else: # Optional: log if already deleted
        #     log.info(f"Cleanup skipped, file already removed: {file_path}")
    except OSError as e:
        log.error(f"Error cleaning up file {file_path}: {e}")

@app.post("/process-zip/", response_class=FileResponse)
async def create_upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a ZIP file containing case folders with PDFs.
    Processes them using Vertex AI (Classification then Extraction).
    Returns an Excel spreadsheet with extracted data, confidence, and reasoning.
    """
    if not file.filename.endswith(".zip"):
        log.error(f"Invalid file type uploaded: {file.filename}. Only .zip files are accepted.")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a ZIP file.")

    log.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")

    temp_zip_path = None # Initialize path variable
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

    if not temp_zip_path: # Check if path was successfully assigned
         raise HTTPException(status_code=500, detail="Failed to create temporary file path.")

    try:
        log.info(f"Starting processing for temporary zip: {temp_zip_path}")
        output_excel_path = process_zip_file(temp_zip_path) # Calls the updated function
        log.info(f"Processing complete. Output Excel at: {output_excel_path}")

        background_tasks.add_task(cleanup_file, output_excel_path)
        background_tasks.add_task(cleanup_file, temp_zip_path)

        return FileResponse(
            path=output_excel_path,
            filename=os.path.basename(output_excel_path),
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
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
    return {"message": "Welcome to the Document Processing API (v2 - Classify then Extract). Use the /process-zip/ endpoint."}

# --- To run the server ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8000