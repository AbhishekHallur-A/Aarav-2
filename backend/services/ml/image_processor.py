"""
Image Processing Service
Handles object detection, OCR, and feature extraction from images
"""

import io
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
from transformers import pipeline, AutoProcessor, AutoModel
import pytesseract
import structlog

from ...utils.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class ImageAnalysisResult:
    """Results from image analysis"""
    objects: List[Dict[str, Any]]
    text: Optional[str] = None
    colors: List[str] = None
    tags: List[str] = None
    embedding: Optional[np.ndarray] = None


class ImageProcessor:
    """
    Advanced image processing with ML models
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ML pipelines
        self.object_detection_pipeline = None
        self.image_classification_pipeline = None
        self.feature_extractor = None
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize ML models for image processing"""
        try:
            logger.info("Initializing image processor...")
            
            # Object detection pipeline
            self.object_detection_pipeline = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Image classification pipeline
            self.image_classification_pipeline = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Feature extractor for embeddings
            self.feature_extractor = AutoProcessor.from_pretrained("microsoft/resnet-50")
            self.feature_model = AutoModel.from_pretrained("microsoft/resnet-50")
            self.feature_model.to(self.device)
            self.feature_model.eval()
            
            self.initialized = True
            logger.info("Image processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize image processor: {e}")
            raise
    
    async def analyze_image(
        self,
        image_data: bytes,
        include_objects: bool = True,
        include_text: bool = True,
        include_colors: bool = True,
        include_embedding: bool = True
    ) -> ImageAnalysisResult:
        """
        Comprehensive image analysis
        
        Args:
            image_data: Raw image bytes
            include_objects: Perform object detection
            include_text: Extract text using OCR
            include_colors: Extract dominant colors
            include_embedding: Generate image embeddings
            
        Returns:
            ImageAnalysisResult with analysis results
        """
        if not self.initialized:
            raise RuntimeError("Image processor not initialized")
        
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            result = ImageAnalysisResult(objects=[])
            
            # Object detection
            if include_objects and self.object_detection_pipeline:
                result.objects = await self._detect_objects(image)
            
            # Text extraction (OCR)
            if include_text:
                result.text = await self._extract_text(image)
            
            # Color extraction
            if include_colors:
                result.colors = await self._extract_colors(image)
            
            # Generate tags from classification
            if self.image_classification_pipeline:
                result.tags = await self._classify_image(image)
            
            # Generate embeddings
            if include_embedding:
                result.embedding = await self._generate_embedding(image)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    async def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        try:
            # Run object detection
            detections = self.object_detection_pipeline(image)
            
            # Process results
            objects = []
            for detection in detections[:10]:  # Top 10 objects
                objects.append({
                    'name': detection['label'],
                    'confidence': detection['score'],
                    'bbox': [
                        detection['box']['xmin'],
                        detection['box']['ymin'],
                        detection['box']['xmax'],
                        detection['box']['ymax']
                    ]
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    async def _extract_text(self, image: Image.Image) -> Optional[str]:
        """Extract text using OCR"""
        try:
            # Enhance image for better OCR
            enhanced_image = ImageEnhance.Contrast(image).enhance(2.0)
            enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(1.5)
            
            # Convert to numpy array for OpenCV processing
            img_array = np.array(enhanced_image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(
                denoised,
                config='--psm 6 --oem 3'  # Page segmentation and OCR engine mode
            )
            
            # Clean extracted text
            cleaned_text = ' '.join(text.split())
            
            return cleaned_text if len(cleaned_text) > 3 else None
            
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return None
    
    async def _extract_colors(self, image: Image.Image) -> List[str]:
        """Extract dominant colors from image"""
        try:
            # Resize image for faster processing
            small_image = image.resize((150, 150))
            
            # Convert to numpy array
            img_array = np.array(small_image)
            
            # Reshape for K-means clustering
            pixels = img_array.reshape(-1, 3)
            
            # Use K-means to find dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = []
            for color in kmeans.cluster_centers_:
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0]), int(color[1]), int(color[2])
                )
                colors.append(hex_color)
            
            return colors
            
        except Exception as e:
            logger.error(f"Error extracting colors: {e}")
            return []
    
    async def _classify_image(self, image: Image.Image) -> List[str]:
        """Classify image to generate tags"""
        try:
            # Run image classification
            classifications = self.image_classification_pipeline(image)
            
            # Extract top labels as tags
            tags = []
            for classification in classifications[:5]:  # Top 5 classifications
                if classification['score'] > 0.1:  # Minimum confidence threshold
                    # Clean label (remove ImageNet class prefixes)
                    label = classification['label'].split(',')[0].strip()
                    tags.append(label.lower())
            
            return tags
            
        except Exception as e:
            logger.error(f"Error in image classification: {e}")
            return []
    
    async def _generate_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Generate image embeddings for similarity search"""
        try:
            # Process image
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.feature_model(**inputs)
                # Use pooled output as embedding
                embedding = outputs.pooler_output.cpu().numpy().flatten()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def find_similar_images(
        self,
        query_embedding: np.ndarray,
        image_database: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar images using embedding similarity"""
        try:
            similarities = []
            
            for item in image_database:
                if 'embedding' in item:
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, item['embedding']) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(item['embedding'])
                    )
                    
                    similarities.append({
                        'item': item,
                        'similarity': similarity
                    })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar images: {e}")
            return []
    
    async def enhance_image_quality(self, image_data: bytes) -> bytes:
        """Enhance image quality for better processing"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Enhance contrast
            enhanced = ImageEnhance.Contrast(image).enhance(1.2)
            
            # Enhance sharpness
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.1)
            
            # Enhance color
            enhanced = ImageEnhance.Color(enhanced).enhance(1.1)
            
            # Save enhanced image
            output = io.BytesIO()
            enhanced.save(output, format=image.format or 'JPEG', quality=95)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image_data
    
    async def health_check(self) -> bool:
        """Check if the image processor is healthy"""
        try:
            if not self.initialized:
                return False
            
            # Create a test image
            test_image = Image.new('RGB', (100, 100), color='red')
            
            # Test object detection
            if self.object_detection_pipeline:
                detections = self.object_detection_pipeline(test_image)
                if not isinstance(detections, list):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False