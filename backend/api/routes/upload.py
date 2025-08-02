"""
Upload API Routes
Handles file uploads for image search and document processing
"""

import asyncio
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel, Field
import structlog

from services.ml.image_processor import ImageProcessor
from services.elasticsearch_client import ElasticsearchManager
from utils.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    content_type: str
    size: int
    upload_time: datetime
    processing_status: str
    url: Optional[str] = None

class ProcessingResult(BaseModel):
    file_id: str
    status: str
    results: Optional[dict] = None
    error: Optional[str] = None

# Dependency injection
async def get_image_processor() -> ImageProcessor:
    """Get image processor instance"""
    processor = ImageProcessor()
    if not processor.initialized:
        await processor.initialize()
    return processor

@router.post("/image", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(..., description="Image file to upload"),
    description: Optional[str] = Form(None, description="Optional description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    image_processor: ImageProcessor = Depends(get_image_processor)
):
    """Upload an image for processing and indexing"""
    try:
        # Validate file type
        if not file.content_type or file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {settings.ALLOWED_IMAGE_TYPES}"
            )
        
        # Validate file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Read file content
        content = await file.read()
        
        # Generate file ID
        file_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        # Process image
        processing_result = await image_processor.process_image(content, file.content_type)
        
        # Store in Elasticsearch
        document = {
            "file_id": file_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size or len(content),
            "upload_time": datetime.now().isoformat(),
            "description": description,
            "tags": tags.split(",") if tags else [],
            "processing_results": processing_result,
            "url": f"/uploads/{file_id}"  # In production, this would be a real URL
        }
        
        # Index document
        success = await ElasticsearchManager.index_document(document, file_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to index document")
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            content_type=file.content_type,
            size=file.size or len(content),
            upload_time=datetime.now(),
            processing_status="completed",
            url=f"/uploads/{file_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/image/{file_id}", response_model=ProcessingResult)
async def get_image_processing_result(
    file_id: str,
    es_manager: ElasticsearchManager = Depends(lambda: ElasticsearchManager)
):
    """Get processing results for an uploaded image"""
    try:
        document = await es_manager.get_document(file_id)
        if not document:
            raise HTTPException(status_code=404, detail="File not found")
        
        return ProcessingResult(
            file_id=file_id,
            status="completed",
            results=document.get("processing_results")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image processing result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/image/{file_id}")
async def delete_uploaded_image(
    file_id: str,
    es_manager: ElasticsearchManager = Depends(lambda: ElasticsearchManager)
):
    """Delete an uploaded image"""
    try:
        success = await es_manager.delete_document(file_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {"message": "File deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/images", response_model=List[UploadResponse])
async def list_uploaded_images(
    limit: int = 10,
    offset: int = 0,
    es_manager: ElasticsearchManager = Depends(lambda: ElasticsearchManager)
):
    """List uploaded images"""
    try:
        # This is a simplified implementation
        # In a real system, you'd query the database for uploaded files
        return []
        
    except Exception as e:
        logger.error(f"Failed to list images: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")