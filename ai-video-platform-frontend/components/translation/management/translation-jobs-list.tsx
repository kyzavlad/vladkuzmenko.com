'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { 
  FiEye, FiDownload, FiTrash, FiSearch, 
  FiFilter, FiClock, FiCheckCircle, FiAlertCircle 
} from 'react-icons/fi';
import { useTranslationContext, TranslationJob } from '../contexts/translation-context';

export const TranslationJobsList: React.FC = () => {
  const router = useRouter();
  const { 
    translationJobs, 
    fetchTranslationJobs, 
    deleteTranslationJob,
    exportTranslatedVideo,
    exportSubtitles,
    isProcessing,
    error 
  } = useTranslationContext();
  
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'processing' | 'completed' | 'failed'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'progress'>('date');
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [downloadOptions, setDownloadOptions] = useState<{
    jobId: string;
    languageId: string;
    show: boolean;
  } | null>(null);
  
  useEffect(() => {
    const loadJobs = async () => {
      setIsLoading(true);
      try {
        await fetchTranslationJobs();
      } catch (err) {
        console.error('Error loading translation jobs:', err);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadJobs();
  }, [fetchTranslationJobs]);
  
  const handleViewJob = (jobId: string, languageId?: string) => {
    const url = languageId 
      ? `/translation?jobId=${jobId}&language=${languageId}`
      : `/translation?jobId=${jobId}`;
    router.push(url);
  };
  
  const handleDeleteJob = async (jobId: string) => {
    try {
      await deleteTranslationJob(jobId);
      setConfirmDelete(null);
    } catch (err) {
      console.error('Error deleting translation job:', err);
    }
  };
  
  const handleExportVideo = async (jobId: string, languageId: string, format: 'mp4' | 'mov' | 'webm' = 'mp4') => {
    try {
      const url = await exportTranslatedVideo(jobId, languageId, format);
      // Trigger download
      window.open(url, '_blank');
    } catch (err) {
      console.error('Error exporting video:', err);
    }
  };
  
  const handleExportSubtitles = async (jobId: string, languageId: string, format: 'srt' | 'vtt' | 'txt' = 'vtt') => {
    try {
      const url = await exportSubtitles(jobId, languageId, format);
      // Trigger download
      window.open(url, '_blank');
    } catch (err) {
      console.error('Error exporting subtitles:', err);
    }
  };
  
  const filteredJobs = translationJobs
    .filter(job => {
      // Apply search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          job.videoId.toLowerCase().includes(query) ||
          job.id.toLowerCase().includes(query)
        );
      }
      return true;
    })
    .filter(job => {
      // Apply status filter
      if (statusFilter === 'all') return true;
      return job.status === statusFilter;
    })
    .sort((a, b) => {
      // Apply sorting
      if (sortBy === 'date') {
        return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
      } else {
        return b.progress - a.progress;
      }
    });
  
  const getStatusBadgeClass = (status: TranslationJob['status']) => {
    switch (status) {
      case 'pending':
        return 'bg-gray-100 text-gray-800';
      case 'processing':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };
  
  const getStatusIcon = (status: TranslationJob['status']) => {
    switch (status) {
      case 'pending':
        return <FiClock />;
      case 'processing':
        return <FiClock className="animate-pulse" />;
      case 'completed':
        return <FiCheckCircle />;
      case 'failed':
        return <FiAlertCircle />;
      default:
        return <FiClock />;
    }
  };
  
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  const getLanguageName = (languageId: string) => {
    const languageMap: Record<string, string> = {
      'en': 'English',
      'es': 'Spanish',
      'fr': 'French',
      'de': 'German',
      'ja': 'Japanese',
      'zh': 'Chinese',
    };
    return languageMap[languageId] || languageId;
  };
  
  if (isLoading) {
    return (
      <div className="w-full bg-white rounded-lg shadow-md p-6 mb-6 flex justify-center items-center h-64">
        <div className="flex flex-col items-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-gray-700">Loading translation jobs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg shadow-md p-6 mb-6">
      <h2 className="text-2xl font-bold mb-4">Translation Jobs</h2>
      
      {error && (
        <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-md">
          {error}
        </div>
      )}
      
      {/* Filters and Search */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="flex items-center relative flex-1 min-w-[250px]">
          <FiSearch className="absolute left-3 text-gray-400" />
          <input
            type="text"
            placeholder="Search jobs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 p-2 border border-gray-300 rounded-md w-full"
          />
        </div>
        
        <div className="flex items-center gap-2">
          <div className="flex items-center">
            <FiFilter className="mr-2 text-gray-500" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as any)}
              className="p-2 border border-gray-300 rounded-md"
            >
              <option value="all">All Statuses</option>
              <option value="pending">Pending</option>
              <option value="processing">Processing</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
          </div>
          
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'date' | 'progress')}
            className="p-2 border border-gray-300 rounded-md"
          >
            <option value="date">Sort by Date</option>
            <option value="progress">Sort by Progress</option>
          </select>
        </div>
      </div>
      
      {/* Jobs List */}
      {filteredJobs.length === 0 ? (
        <div className="text-center py-10 text-gray-500">
          <p className="text-lg">No translation jobs found</p>
          <p className="text-sm mt-2">
            {searchQuery || statusFilter !== 'all' 
              ? 'Try adjusting your filters' 
              : 'Start by translating a video'}
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Video
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Languages
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredJobs.map(job => (
                <tr key={job.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-10 w-16 rounded overflow-hidden bg-gray-100">
                        {job.thumbnailUrl ? (
                          <img 
                            src={job.thumbnailUrl} 
                            alt={`Thumbnail for ${job.videoId}`} 
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <div className="flex items-center justify-center h-full w-full text-gray-400">
                            No Image
                          </div>
                        )}
                      </div>
                      <div className="ml-4">
                        <div className="text-sm font-medium text-gray-900">
                          {job.videoId}
                        </div>
                        <div className="text-sm text-gray-500">
                          {job.videoDuration ? `${Math.floor(job.videoDuration / 60)}:${(job.videoDuration % 60).toString().padStart(2, '0')}` : 'N/A'}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">
                      From: <span className="font-medium">{getLanguageName(job.sourceLanguage)}</span>
                    </div>
                    <div className="text-sm text-gray-500">
                      To: {job.targetLanguages.map(lang => getLanguageName(lang)).join(', ')}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusBadgeClass(job.status)}`}>
                        <span className="mr-1">{getStatusIcon(job.status)}</span>
                        {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                      </span>
                      {job.status === 'processing' && (
                        <div className="ml-3 w-16">
                          <div className="h-2 bg-gray-200 rounded-full">
                            <div 
                              className="h-2 bg-blue-600 rounded-full" 
                              style={{ width: `${job.progress}%` }}
                            ></div>
                          </div>
                          <div className="text-xs text-gray-500 mt-1">{job.progress}%</div>
                        </div>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatDate(job.updatedAt)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex justify-end space-x-2">
                      <button
                        className="text-blue-600 hover:text-blue-900 p-1"
                        onClick={() => handleViewJob(job.id)}
                      >
                        <FiEye size={18} />
                      </button>
                      
                      {job.status === 'completed' && (
                        <div className="relative">
                          <button
                            className="text-green-600 hover:text-green-900 p-1"
                            onClick={() => setDownloadOptions({ 
                              jobId: job.id, 
                              languageId: job.targetLanguages[0],
                              show: true 
                            })}
                          >
                            <FiDownload size={18} />
                          </button>
                          
                          {downloadOptions?.jobId === job.id && downloadOptions.show && (
                            <div className="absolute right-0 mt-2 w-64 bg-white rounded-md shadow-lg z-10 border border-gray-200">
                              <div className="p-3 border-b">
                                <h3 className="font-medium">Download Options</h3>
                                <p className="text-xs text-gray-500">
                                  Select language and format
                                </p>
                              </div>
                              <div className="p-3">
                                <div className="mb-3">
                                  <label className="block text-xs text-gray-700 mb-1">
                                    Language
                                  </label>
                                  <select
                                    className="w-full p-1.5 border border-gray-300 rounded-md text-sm"
                                    value={downloadOptions.languageId}
                                    onChange={(e) => setDownloadOptions({
                                      ...downloadOptions,
                                      languageId: e.target.value
                                    })}
                                  >
                                    {job.targetLanguages.map(lang => (
                                      <option key={lang} value={lang}>
                                        {getLanguageName(lang)}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                                
                                <div className="space-y-2">
                                  <button
                                    className="w-full text-left text-sm p-1.5 hover:bg-blue-50 rounded flex items-center"
                                    onClick={() => handleExportVideo(job.id, downloadOptions.languageId, 'mp4')}
                                  >
                                    <FiDownload className="mr-2" />
                                    Download Video (MP4)
                                  </button>
                                  <button
                                    className="w-full text-left text-sm p-1.5 hover:bg-blue-50 rounded flex items-center"
                                    onClick={() => handleExportSubtitles(job.id, downloadOptions.languageId, 'vtt')}
                                  >
                                    <FiDownload className="mr-2" />
                                    Download Subtitles (VTT)
                                  </button>
                                  <button
                                    className="w-full text-left text-sm p-1.5 hover:bg-blue-50 rounded flex items-center"
                                    onClick={() => handleExportSubtitles(job.id, downloadOptions.languageId, 'srt')}
                                  >
                                    <FiDownload className="mr-2" />
                                    Download Subtitles (SRT)
                                  </button>
                                </div>
                              </div>
                              <div className="p-3 border-t flex justify-end">
                                <button
                                  className="text-xs text-gray-600 hover:text-gray-900"
                                  onClick={() => setDownloadOptions(null)}
                                >
                                  Close
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      
                      <div className="relative">
                        <button
                          className="text-red-600 hover:text-red-900 p-1"
                          onClick={() => setConfirmDelete(job.id)}
                        >
                          <FiTrash size={18} />
                        </button>
                        
                        {confirmDelete === job.id && (
                          <div className="absolute right-0 mt-2 w-64 bg-white rounded-md shadow-lg z-10 border border-gray-200">
                            <div className="p-3">
                              <h3 className="font-medium text-red-600">Delete Job?</h3>
                              <p className="text-sm mt-1 text-gray-700">
                                This action cannot be undone.
                              </p>
                              <div className="mt-3 flex justify-end space-x-2">
                                <button
                                  className="px-3 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50"
                                  onClick={() => setConfirmDelete(null)}
                                >
                                  Cancel
                                </button>
                                <button
                                  className="px-3 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700"
                                  onClick={() => handleDeleteJob(job.id)}
                                >
                                  Delete
                                </button>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      {downloadOptions && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-30 z-10"
          onClick={() => setDownloadOptions(null)}
        ></div>
      )}
      
      {confirmDelete && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-30 z-10"
          onClick={() => setConfirmDelete(null)}
        ></div>
      )}
    </div>
  );
};

export default TranslationJobsList; 