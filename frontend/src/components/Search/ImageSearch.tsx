import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PhotoIcon,
  XMarkIcon,
  CameraIcon,
  ArrowUpTrayIcon,
  ExclamationTriangleIcon,
  MagnifyingGlassIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';
import { useDropzone } from 'react-dropzone';

// Types
interface ImageSearchProps {
  isOpen: boolean;
  onClose: () => void;
  onImageUpload: (file: File) => void;
}

interface UploadedImage {
  file: File;
  preview: string;
  analysisResult?: ImageAnalysisResult;
}

interface ImageAnalysisResult {
  objects: Array<{
    name: string;
    confidence: number;
    bbox?: number[];
  }>;
  text?: string;
  colors: string[];
  tags: string[];
}

const ImageSearch: React.FC<ImageSearchProps> = ({
  isOpen,
  onClose,
  onImageUpload,
}) => {
  const [uploadedImage, setUploadedImage] = useState<UploadedImage | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useCamera, setUseCamera] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Dropzone configuration
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      handleFileUpload(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp', '.bmp']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
    onDropRejected: (rejectedFiles) => {
      const rejection = rejectedFiles[0];
      if (rejection.errors[0].code === 'file-too-large') {
        setError('File size must be less than 10MB');
      } else if (rejection.errors[0].code === 'file-invalid-type') {
        setError('Please upload a valid image file');
      } else {
        setError('Invalid file');
      }
    }
  });

  const handleFileUpload = async (file: File) => {
    try {
      setError(null);
      setIsAnalyzing(true);

      // Create preview URL
      const preview = URL.createObjectURL(file);
      
      const imageData: UploadedImage = {
        file,
        preview,
      };

      // Analyze image
      const analysisResult = await analyzeImage(file);
      imageData.analysisResult = analysisResult;

      setUploadedImage(imageData);
      setIsAnalyzing(false);
    } catch (error) {
      console.error('Error uploading image:', error);
      setError('Failed to process image');
      setIsAnalyzing(false);
    }
  };

  const analyzeImage = async (file: File): Promise<ImageAnalysisResult> => {
    // Simulate image analysis - in production, this would call your ML API
    return new Promise((resolve) => {
      setTimeout(() => {
        // Mock analysis result
        const mockResult: ImageAnalysisResult = {
          objects: [
            { name: 'person', confidence: 0.95 },
            { name: 'building', confidence: 0.87 },
            { name: 'car', confidence: 0.73 },
          ],
          text: 'Sample extracted text from image',
          colors: ['#FF5733', '#33C4FF', '#75FF33'],
          tags: ['outdoor', 'urban', 'daytime', 'street'],
        };
        resolve(mockResult);
      }, 2000);
    });
  };

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      
      setStream(mediaStream);
      setUseCamera(true);
      setError(null);
    } catch (error) {
      console.error('Error accessing camera:', error);
      setError('Could not access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setUseCamera(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      if (context) {
        context.drawImage(video, 0, 0);
        
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            handleFileUpload(file);
            stopCamera();
          }
        }, 'image/jpeg', 0.9);
      }
    }
  };

  const handleSearch = () => {
    if (uploadedImage) {
      onImageUpload(uploadedImage.file);
      onClose();
    }
  };

  const handleClose = () => {
    stopCamera();
    if (uploadedImage) {
      URL.revokeObjectURL(uploadedImage.preview);
    }
    setUploadedImage(null);
    setError(null);
    onClose();
  };

  const removeImage = () => {
    if (uploadedImage) {
      URL.revokeObjectURL(uploadedImage.preview);
    }
    setUploadedImage(null);
    setError(null);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50 backdrop-blur-sm"
        onClick={handleClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto p-8 bg-white dark:bg-gray-800 rounded-2xl shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Close Button */}
          <button
            onClick={handleClose}
            className="absolute top-4 right-4 p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors z-10"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>

          {/* Header */}
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Image Search
            </h2>
            <p className="text-gray-600 dark:text-gray-300">
              Upload an image or take a photo to find similar images
            </p>
          </div>

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg"
            >
              <div className="flex items-center">
                <ExclamationTriangleIcon className="w-5 h-5 text-red-500 mr-2" />
                <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
              </div>
            </motion.div>
          )}

          {/* Camera View */}
          {useCamera && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mb-6"
            >
              <div className="relative bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full h-64 object-cover"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <button
                    onClick={capturePhoto}
                    className="w-16 h-16 bg-white bg-opacity-20 border-4 border-white rounded-full flex items-center justify-center hover:bg-opacity-30 transition-all"
                  >
                    <div className="w-8 h-8 bg-white rounded-full" />
                  </button>
                </div>
                <button
                  onClick={stopCamera}
                  className="absolute top-4 right-4 p-2 bg-black bg-opacity-50 text-white rounded-lg hover:bg-opacity-70 transition-opacity"
                >
                  <XMarkIcon className="w-5 h-5" />
                </button>
              </div>
            </motion.div>
          )}

          {/* Upload Area or Image Display */}
          {!useCamera && !uploadedImage && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6"
            >
              <div
                {...getRootProps()}
                className={`
                  relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
                  transition-all duration-200
                  ${isDragActive
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500'
                  }
                `}
              >
                <input {...getInputProps()} />
                <PhotoIcon className="w-16 h-16 mx-auto text-gray-400 dark:text-gray-500 mb-4" />
                
                {isDragActive ? (
                  <p className="text-blue-600 dark:text-blue-400 font-medium">
                    Drop the image here...
                  </p>
                ) : (
                  <div className="space-y-2">
                    <p className="text-gray-600 dark:text-gray-300 font-medium">
                      Drag and drop an image here, or click to select
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Supports JPG, PNG, GIF, WebP (max 10MB)
                    </p>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Uploaded Image Display */}
          {uploadedImage && !useCamera && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mb-6"
            >
              <div className="relative">
                <img
                  src={uploadedImage.preview}
                  alt="Uploaded"
                  className="w-full max-h-64 object-contain bg-gray-100 dark:bg-gray-700 rounded-lg"
                />
                <button
                  onClick={removeImage}
                  className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                >
                  <TrashIcon className="w-4 h-4" />
                </button>
              </div>

              {/* Analysis Results */}
              {uploadedImage.analysisResult && !isAnalyzing && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 space-y-4"
                >
                  {/* Detected Objects */}
                  {uploadedImage.analysisResult.objects.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Detected Objects:
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {uploadedImage.analysisResult.objects.map((object, index) => (
                          <span
                            key={index}
                            className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 text-sm rounded-full"
                          >
                            {object.name} ({Math.round(object.confidence * 100)}%)
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Tags */}
                  {uploadedImage.analysisResult.tags.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Tags:
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {uploadedImage.analysisResult.tags.map((tag, index) => (
                          <span
                            key={index}
                            className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 text-sm rounded-full"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Extracted Text */}
                  {uploadedImage.analysisResult.text && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Extracted Text:
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                        {uploadedImage.analysisResult.text}
                      </p>
                    </div>
                  )}
                </motion.div>
              )}

              {/* Loading Animation */}
              {isAnalyzing && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-4 text-center"
                >
                  <div className="inline-flex items-center space-x-2 text-gray-600 dark:text-gray-300">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full"
                    />
                    <span className="text-sm">Analyzing image...</span>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}

          {/* Action Buttons */}
          <div className="space-y-4">
            {/* Camera/Upload Toggle */}
            {!uploadedImage && (
              <div className="flex space-x-4">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-1 flex items-center justify-center px-4 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                  <ArrowUpTrayIcon className="w-5 h-5 mr-2" />
                  Upload Image
                </button>
                
                <button
                  onClick={startCamera}
                  className="flex-1 flex items-center justify-center px-4 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
                >
                  <CameraIcon className="w-5 h-5 mr-2" />
                  Use Camera
                </button>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileUpload(file);
                  }}
                  className="hidden"
                />
              </div>
            )}

            {/* Main Actions */}
            <div className="flex space-x-4">
              <button
                onClick={handleClose}
                className="flex-1 px-4 py-2 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                Cancel
              </button>
              
              {uploadedImage && (
                <button
                  onClick={handleSearch}
                  disabled={isAnalyzing}
                  className="flex-1 flex items-center justify-center px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <MagnifyingGlassIcon className="w-5 h-5 mr-2" />
                  Search Similar Images
                </button>
              )}
            </div>
          </div>

          {/* Tips */}
          <div className="mt-6 text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              For best results, use clear images with good lighting
            </p>
          </div>

          {/* Hidden Canvas for Camera Capture */}
          <canvas ref={canvasRef} className="hidden" />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default ImageSearch;