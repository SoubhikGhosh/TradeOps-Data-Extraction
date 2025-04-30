# main.py
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile

from utils import log, setup_logger
from processing import process_zip_file
from config import TEMP_DIR

# Ensure temp processing directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize logger (in case utils hasn't been imported elsewhere first)
setup_logger()

app = FastAPI(title="Document Processing Service", version="1.0.0")

def cleanup_file(file_path: str):
    """Background task to delete a file."""
    try:
        os.remove(file_path)
        log.info(f"Cleaned up temporary file: {file_path}")
    except OSError as e:
        log.error(f"Error cleaning up file {file_path}: {e}")

@app.post("/process-zip/", response_class=FileResponse)
async def create_upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a ZIP file containing case folders with PDFs,
    processes them using Vertex AI, and returns an Excel spreadsheet
    with extracted data, confidence, and reasoning.
    """
    if not file.filename.endswith(".zip"):
        log.error(f"Invalid file type uploaded: {file.filename}. Only .zip files are accepted.")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a ZIP file.")

    log.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")

    # Save the uploaded zip file temporarily
    # Using NamedTemporaryFile for automatic cleanup is safer if processing fails early
    try:
        # suffix=".zip" helps identify the file type if needed
        # delete=False is important because process_zip_file needs to open it by path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=TEMP_DIR) as temp_zip_file:
            shutil.copyfileobj(file.file, temp_zip_file)
            temp_zip_path = temp_zip_file.name # Get the path
        log.info(f"Saved uploaded zip file temporarily to: {temp_zip_path}")

    except Exception as e:
         log.exception(f"Failed to save uploaded file {file.filename}: {e}")
         raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        # Ensure the file object provided by FastAPI is closed
        await file.close()


    try:
        # --- Trigger the main processing logic ---
        log.info(f"Starting processing for temporary zip: {temp_zip_path}")
        output_excel_path = process_zip_file(temp_zip_path)
        log.info(f"Processing complete. Output Excel at: {output_excel_path}")

        # --- Return the Excel file ---
        # Add the output excel file to background tasks for cleanup after response sent
        background_tasks.add_task(cleanup_file, output_excel_path)
        # Also add the temporary input zip file for cleanup
        background_tasks.add_task(cleanup_file, temp_zip_path)

        return FileResponse(
            path=output_excel_path,
            filename=os.path.basename(output_excel_path), # Use the generated filename
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except ValueError as ve: # Specific errors like bad zip, no folders
         log.error(f"Value Error during processing: {ve}")
         background_tasks.add_task(cleanup_file, temp_zip_path) # Cleanup input zip on error
         raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re: # Specific errors like Excel saving failure
         log.error(f"Runtime Error during processing: {re}")
         background_tasks.add_task(cleanup_file, temp_zip_path) # Cleanup input zip on error
         raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        log.exception(f"An unexpected error occurred during processing zip file {temp_zip_path}: {e}")
        background_tasks.add_task(cleanup_file, temp_zip_path) # Cleanup input zip on error
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@app.get("/")
async def root():
    return {"message": "Welcome to the Document Processing API. Use the /process-zip/ endpoint to upload files."}

# --- To run the server (e.g., using uvicorn) ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8000