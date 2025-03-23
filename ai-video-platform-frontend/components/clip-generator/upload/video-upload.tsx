'use client';

import React, { useState, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useClipContext } from '../contexts/clip-context';
import { 
  FiUpload, FiLink, FiX, FiCheck, FiAlertCircle, 
  FiYoutube, FiVideo, FiTwitch, FiGrid, FiCheckSquare,
  FiSquare, FiPlusCircle, FiLoader
} from 'react-icons/fi';

type UploadMethod = 'file' | 'url';
type ContentCategory = 'technology' | 'education' | 'entertainment' | 'gaming' | 'howto' | 'vlog' | 'other';
type PlatformPreset = 'youtube' | 'twitch' | 'vimeo' | 'tiktok' | 'custom';

interface ContentCategoryOption {
  id: ContentCategory;
  label: string;
  description: string;
  icon: React.ReactNode;
}

interface PlatformPresetOption {
  id: PlatformPreset;
  label: string;
  icon: React.ReactNode;
  description: string;
}

interface VideoFile {
  id: string;
  file: File;
  name: string;
  size: number;
  progress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  error?: string;
  selected: boolean;
}

const VideoUpload: React.FC = () => {
  const { setUploadedVideo, setGeneratedClips } = useClipContext();
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [uploadMethod, setUploadMethod] = useState<UploadMethod>('file');
  const [videoUrl, setVideoUrl] = useState('');
  const [selectedPlatform, setSelectedPlatform] = useState<PlatformPreset>('youtube');
  const [selectedCategory, setSelectedCategory] = useState<ContentCategory>('entertainment');
  const [files, setFiles] = useState<VideoFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLibrary, setShowLibrary] = useState(true);
  const [selectMode, setSelectMode] = useState(false);
  const [allVideosSelected, setAllVideosSelected] = useState(false);

  // Content category options
  const contentCategories: ContentCategoryOption[] = [
    { 
      id: 'technology', 
      label: 'Technology', 
      description: 'Tech reviews, gadgets, software tutorials', 
      icon: <span className="text-blue-500">💻</span> 
    },
    { 
      id: 'education', 
      label: 'Education', 
      description: 'Lectures, explainers, educational content', 
      icon: <span className="text-green-500">🎓</span> 
    },
    { 
      id: 'entertainment', 
      label: 'Entertainment', 
      description: 'Shows, comedy, reactions, vlogs', 
      icon: <span className="text-purple-500">🎭</span> 
    },
    { 
      id: 'gaming', 
      label: 'Gaming', 
      description: 'Gameplay, reviews, walkthroughs', 
      icon: <span className="text-red-500">🎮</span> 
    },
    { 
      id: 'howto', 
      label: 'How-to & DIY', 
      description: 'Tutorials, guides, DIY projects', 
      icon: <span className="text-yellow-500">🔧</span> 
    },
    { 
      id: 'vlog', 
      label: 'Vlogs', 
      description: 'Personal vlogs, day-in-life content', 
      icon: <span className="text-pink-500">📹</span> 
    },
    { 
      id: 'other', 
      label: 'Other', 
      description: 'Other content types', 
      icon: <span className="text-gray-500">📄</span> 
    }
  ];

  // Platform preset options
  const platformPresets: PlatformPresetOption[] = [
    { 
      id: 'youtube', 
      label: 'YouTube', 
      icon: <FiYoutube className="text-red-600" />,
      description: 'Optimized for YouTube Shorts (9:16 vertical format)'
    },
    { 
      id: 'twitch', 
      label: 'Twitch', 
      icon: <FiTwitch className="text-purple-600" />,
      description: 'Highlights key moments with transitions'
    },
    { 
      id: 'tiktok', 
      label: 'TikTok', 
      icon: <span className="text-black dark:text-white">📱</span>,
      description: 'Fast-paced vertical clips with text overlays'
    },
    { 
      id: 'vimeo', 
      label: 'Vimeo', 
      icon: <FiVideo className="text-blue-600" />,
      description: 'Professional-looking clips with subtle transitions'
    },
    { 
      id: 'custom', 
      label: 'Custom', 
      icon: <FiGrid className="text-gray-600" />,
      description: 'Customize settings manually'
    }
  ];

  // Handle file upload
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).map(file => ({
        id: Math.random().toString(36).substring(2, 9),
        file,
        name: file.name,
        size: file.size,
        progress: 0,
        status: 'pending' as const,
        selected: true
      }));
      
      setFiles(prev => [...prev, ...newFiles]);
      e.target.value = ''; // Reset input
    }
  };

  // Handle URL import
  const handleUrlImport = () => {
    if (!videoUrl) {
      setError('Please enter a valid URL');
      return;
    }
    
    // Validate URL
    try {
      new URL(videoUrl);
    } catch (e) {
      setError('Please enter a valid URL');
      return;
    }
    
    setError(null);
    // TODO: Implement URL import logic
    
    // For now, just simulate a successful import
    const fakeFile: VideoFile = {
      id: Math.random().toString(36).substring(2, 9),
      file: new File([], videoUrl.split('/').pop() || 'video.mp4'),
      name: videoUrl.split('/').pop() || 'Imported Video',
      size: 0,
      progress: 100,
      status: 'success',
      selected: true
    };
    
    setFiles(prev => [...prev, fakeFile]);
    setVideoUrl('');
  };

  // Handle drag events
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  // Handle drop event
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files)
        .filter(file => file.type.startsWith('video/'))
        .map(file => ({
          id: Math.random().toString(36).substring(2, 9),
          file,
          name: file.name,
          size: file.size,
          progress: 0,
          status: 'pending' as const,
          selected: true
        }));
      
      if (droppedFiles.length > 0) {
        setFiles(prev => [...prev, ...droppedFiles]);
      } else {
        setError('Please upload video files only (mp4, mov, etc.)');
      }
    }
  }, []);

  // Simulate file upload
  const uploadFiles = async () => {
    if (files.length === 0) {
      setError('Please select at least one video to upload');
      return;
    }
    
    setUploading(true);
    setError(null);
    
    // Update status for all files to uploading
    setFiles(prev => prev.map(file => ({
      ...file,
      status: file.status === 'pending' ? 'uploading' : file.status
    })));
    
    // Process each file with artificial delay
    for (const file of files) {
      if (file.status !== 'uploading') continue;
      
      // Simulate upload progress
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setFiles(prev => prev.map(f => 
          f.id === file.id ? { ...f, progress } : f
        ));
      }
      
      // Mark as success
      setFiles(prev => prev.map(f => 
        f.id === file.id ? { ...f, status: 'success', progress: 100 } : f
      ));
      
      // Set uploaded video in context
      setUploadedVideo(file.file);
    }
    
    setUploading(false);
  };

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(file => file.id !== id));
  };

  const toggleVideoSelection = (id: string) => {
    setFiles(prev => prev.map(file => 
      file.id === id ? { ...file, selected: !file.selected } : file
    ));
  };

  const toggleAllVideos = () => {
    setAllVideosSelected(!allVideosSelected);
    setFiles(prev => prev.map(file => ({ ...file, selected: !allVideosSelected })));
  };

  const generateClips = () => {
    const selectedFiles = files.filter(file => file.selected);
    if (selectedFiles.length === 0) {
      setError('Please select at least one video to generate clips from');
      return;
    }
    
    // Set the first selected file as the uploaded video
    setUploadedVideo(selectedFiles[0].file);
    
    // Navigate to the clip generation page
    router.push('/clip-generator/generate');
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Upload Video</h1>
        
        {/* Upload Method Selection */}
        <div className="mb-8">
          <div className="flex space-x-4">
            <button
              className={`flex-1 p-4 rounded-lg border ${
                uploadMethod === 'file'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
              onClick={() => setUploadMethod('file')}
            >
              <FiUpload className="w-6 h-6 mx-auto mb-2" />
              <span className="block text-center">Upload File</span>
            </button>
            <button
              className={`flex-1 p-4 rounded-lg border ${
                uploadMethod === 'url'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
              onClick={() => setUploadMethod('url')}
            >
              <FiLink className="w-6 h-6 mx-auto mb-2" />
              <span className="block text-center">Import from URL</span>
            </button>
          </div>
        </div>
        
        {/* File Upload Area */}
        {uploadMethod === 'file' && (
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center ${
              dragActive
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-300 dark:border-gray-600'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              className="hidden"
            />
            <FiUpload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p className="text-gray-600 dark:text-gray-400 mb-2">
              Drag and drop your video files here, or click to select files
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-500">
              Supported formats: MP4, MOV, AVI, MKV
            </p>
          </div>
        )}
        
        {/* URL Import Form */}
        {uploadMethod === 'url' && (
          <div className="space-y-4">
            <div className="flex space-x-4">
              <input
                type="url"
                placeholder="Enter video URL"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                className="flex-1 p-3 border border-gray-300 dark:border-gray-600 rounded-lg"
              />
              <button
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                onClick={handleUrlImport}
              >
                Import
              </button>
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-500">
              Supported platforms: YouTube, Vimeo, Twitch
            </p>
          </div>
        )}
        
        {/* File List */}
        {files.length > 0 && (
          <div className="mt-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Selected Files</h2>
              <div className="flex space-x-2">
                <button
                  className="px-4 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg"
                  onClick={() => setFiles([])}
                >
                  Clear All
                </button>
                <button
                  className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                  onClick={uploadFiles}
                  disabled={uploading}
                >
                  {uploading ? (
                    <>
                      <FiLoader className="inline-block animate-spin mr-2" />
                      Uploading...
                    </>
                  ) : (
                    'Upload Files'
                  )}
                </button>
              </div>
            </div>
            
            <div className="space-y-4">
              {files.map(file => (
                <div
                  key={file.id}
                  className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex items-center space-x-4">
                    <input
                      type="checkbox"
                      checked={file.selected}
                      onChange={() => toggleVideoSelection(file.id)}
                      className="w-5 h-5 text-blue-600 rounded border-gray-300"
                    />
                    <div>
                      <p className="font-medium">{file.name}</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {formatFileSize(file.size)}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    {file.status === 'uploading' && (
                      <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-600 transition-all duration-300"
                          style={{ width: `${file.progress}%` }}
                        />
                      </div>
                    )}
                    
                    {file.status === 'success' && (
                      <FiCheck className="text-green-500" />
                    )}
                    
                    {file.status === 'error' && (
                      <FiAlertCircle className="text-red-500" />
                    )}
                    
                    <button
                      className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                      onClick={() => removeFile(file.id)}
                    >
                      <FiX />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400">
            {error}
          </div>
        )}
        
        {/* Next Step Button */}
        {files.some(file => file.status === 'success') && (
          <div className="mt-8 text-center">
            <button
              className="px-8 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              onClick={generateClips}
            >
              Continue to Clip Generation
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoUpload; 