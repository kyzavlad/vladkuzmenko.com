'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { FiDownload, FiShare2, FiCopy, FiScissors } from 'react-icons/fi';

interface ExportOptionsProps {
  videoId: string;
  videoTitle: string;
  videoThumbnail: string;
  duration: number;
}

const ExportOptions: React.FC<ExportOptionsProps> = ({
  videoId,
  videoTitle,
  videoThumbnail,
  duration
}) => {
  const router = useRouter();
  const [showShareOptions, setShowShareOptions] = useState(false);
  const [copied, setCopied] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState('mp4');
  const [selectedQuality, setSelectedQuality] = useState('1080p');
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);

  const videoUrl = `https://ai-video-platform.com/videos/${videoId}`;

  const handleCopyLink = () => {
    navigator.clipboard.writeText(videoUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    // Simulate download progress
    setDownloadProgress(0);
    const interval = setInterval(() => {
      setDownloadProgress(prev => {
        if (prev === null || prev >= 100) {
          clearInterval(interval);
          return null;
        }
        return prev + 10;
      });
    }, 500);
  };

  const handleGenerateClips = () => {
    // Navigate to clip generator with the current video pre-selected
    router.push(`/clip-generator?videoId=${videoId}`);
  };

  const formatOptions = [
    { value: 'mp4', label: 'MP4' },
    { value: 'mov', label: 'MOV' },
    { value: 'webm', label: 'WebM' }
  ];

  const qualityOptions = [
    { value: '720p', label: '720p HD' },
    { value: '1080p', label: '1080p Full HD' },
    { value: '2k', label: '2K QHD' },
    { value: '4k', label: '4K UHD' }
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden">
      <div className="p-6">
        <h2 className="text-2xl font-bold mb-4">Export Options</h2>
        
        <div className="flex flex-col md:flex-row gap-6">
          {/* Video Preview */}
          <div className="w-full md:w-1/3">
            <div className="bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden aspect-video relative">
              <img 
                src={videoThumbnail} 
                alt={videoTitle}
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                {Math.floor(duration / 60)}:{(duration % 60).toString().padStart(2, '0')}
              </div>
            </div>
            <h3 className="font-medium mt-2 text-gray-900 dark:text-white">{videoTitle}</h3>
            <div className="text-sm text-gray-500 dark:text-gray-400">ID: {videoId}</div>
            
            <div className="mt-4 space-y-3">
              <button
                onClick={() => setShowShareOptions(!showShareOptions)}
                className="flex items-center justify-center w-full py-2 px-4 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <FiShare2 className="mr-2" />
                Share
              </button>
              
              {showShareOptions && (
                <div className="p-3 bg-gray-100 dark:bg-gray-700 rounded-lg">
                  <div className="flex">
                    <input
                      type="text"
                      value={videoUrl}
                      readOnly
                      className="flex-1 text-sm p-2 border border-gray-300 dark:border-gray-600 rounded-l-lg bg-white dark:bg-gray-800"
                    />
                    <button
                      onClick={handleCopyLink}
                      className="p-2 bg-blue-600 text-white rounded-r-lg hover:bg-blue-700"
                    >
                      <FiCopy />
                    </button>
                  </div>
                  <div className="mt-2 flex justify-end text-xs">
                    {copied && <span className="text-green-600">Copied to clipboard!</span>}
                  </div>
                </div>
              )}
              
              <button
                onClick={handleGenerateClips}
                className="flex items-center justify-center w-full py-2 px-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium transition-colors"
              >
                <FiScissors className="mr-2" />
                Generate Clips
              </button>
              <div className="text-xs text-center text-gray-500 dark:text-gray-400">
                Create short-form content from this video
              </div>
            </div>
          </div>
          
          {/* Export Settings */}
          <div className="w-full md:w-2/3">
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Format
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {formatOptions.map(option => (
                    <button
                      key={option.value}
                      className={`py-2 px-3 rounded-lg text-sm font-medium ${
                        selectedFormat === option.value
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                      }`}
                      onClick={() => setSelectedFormat(option.value)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Quality
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {qualityOptions.map(option => (
                    <button
                      key={option.value}
                      className={`py-2 px-3 rounded-lg text-sm font-medium ${
                        selectedQuality === option.value
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                      }`}
                      onClick={() => setSelectedQuality(option.value)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Additional Options
                </label>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="includeSubtitles"
                      className="h-4 w-4 text-blue-600 rounded border-gray-300"
                    />
                    <label htmlFor="includeSubtitles" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                      Include subtitles (SRT)
                    </label>
                  </div>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="optimizeWeb"
                      className="h-4 w-4 text-blue-600 rounded border-gray-300"
                      defaultChecked
                    />
                    <label htmlFor="optimizeWeb" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                      Optimize for web
                    </label>
                  </div>
                </div>
              </div>
              
              <div>
                <button
                  onClick={handleDownload}
                  disabled={downloadProgress !== null}
                  className="w-full flex items-center justify-center py-3 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg font-medium transition-colors"
                >
                  <FiDownload className="mr-2" />
                  {downloadProgress !== null ? 'Downloading...' : 'Download Video'}
                </button>
                
                {downloadProgress !== null && (
                  <div className="mt-2">
                    <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-600"
                        style={{ width: `${downloadProgress}%` }}
                      ></div>
                    </div>
                    <div className="mt-1 text-xs text-right text-gray-500 dark:text-gray-400">
                      {downloadProgress}% complete
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExportOptions; 