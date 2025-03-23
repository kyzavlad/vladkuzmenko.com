'use client';

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiGrid, 
  FiList, 
  FiFilter, 
  FiSearch, 
  FiChevronDown, 
  FiEdit2, 
  FiTrash2, 
  FiInfo,
  FiX,
  FiCheck,
  FiAlertCircle,
  FiClock,
  FiArrowUp,
  FiArrowDown,
  FiFileText
} from 'react-icons/fi';
import { useMediaContext, MediaFile } from '../contexts/media-context';
import MediaGrid from './media-grid';
import MediaList from './media-list';
import MediaDetails from './media-details';
import { formatDate, formatDuration, formatFileSize } from '../../../lib/utils/formatters';

export default function MediaLibrary() {
  const { 
    mediaFiles, 
    selectedFiles,
    isLoading, 
    error, 
    viewMode, 
    sortBy, 
    sortDirection,
    filterOptions,
    searchQuery,
    fetchMediaFiles,
    setViewMode,
    setSortBy,
    setSortDirection,
    setFilterOptions,
    setSearchQuery,
    selectFile,
    deselectFile,
    selectAllFiles,
    deselectAllFiles,
    deleteFiles
  } = useMediaContext();
  
  const [showFilters, setShowFilters] = useState(false);
  const [detailsFile, setDetailsFile] = useState<MediaFile | null>(null);
  
  useEffect(() => {
    fetchMediaFiles();
  }, [fetchMediaFiles]);
  
  const handleSelectFile = (file: MediaFile) => {
    if (file.isSelected) {
      deselectFile(file.id);
    } else {
      selectFile(file.id);
    }
  };
  
  const handleDetailsClose = () => {
    setDetailsFile(null);
  };
  
  const handleDeleteSelected = async () => {
    if (selectedFiles.length === 0) return;
    
    // Ask for confirmation
    if (window.confirm(`Are you sure you want to delete ${selectedFiles.length} file(s)?`)) {
      const selectedIds = selectedFiles.map(file => file.id);
      await deleteFiles(selectedIds);
    }
  };
  
  const toggleSortDirection = () => {
    setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
  };
  
  const renderSortOptions = () => (
    <div className="relative">
      <div className="flex items-center space-x-2">
        <select
          className="bg-neutral-400 border border-neutral-300 rounded-md p-2 text-sm text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as any)}
        >
          <option value="date">Date Added</option>
          <option value="name">Name</option>
          <option value="size">Size</option>
          <option value="duration">Duration</option>
        </select>
        <button
          className="p-2 bg-neutral-400 rounded-md text-neutral-100 hover:bg-neutral-300 transition-colors"
          onClick={toggleSortDirection}
          title={sortDirection === 'asc' ? 'Ascending' : 'Descending'}
        >
          {sortDirection === 'asc' ? <FiArrowUp size={16} /> : <FiArrowDown size={16} />}
        </button>
      </div>
    </div>
  );
  
  const renderFilterOptions = () => (
    <AnimatePresence>
      {showFilters && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2 }}
          className="bg-neutral-400 p-4 rounded-lg mt-3 space-y-4 shadow-elevation-1"
        >
          <div>
            <h3 className="font-medium text-sm mb-2 text-neutral-100">Status</h3>
            <div className="flex flex-wrap gap-2">
              {['processing', 'ready', 'error'].map((status) => (
                <button
                  key={status}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    filterOptions.status.includes(status as any)
                      ? 'bg-primary text-white'
                      : 'bg-neutral-300 text-neutral-100 hover:bg-neutral-200'
                  }`}
                  onClick={() => {
                    const newStatus = filterOptions.status.includes(status as any)
                      ? filterOptions.status.filter(s => s !== status)
                      : [...filterOptions.status, status];
                    
                    setFilterOptions({ status: newStatus as any[] });
                  }}
                >
                  {status === 'processing' && <FiClock size={12} className="inline mr-1" />}
                  {status === 'ready' && <FiCheck size={12} className="inline mr-1" />}
                  {status === 'error' && <FiAlertCircle size={12} className="inline mr-1" />}
                  {status.charAt(0).toUpperCase() + status.slice(1)}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <h3 className="font-medium text-sm mb-2 text-neutral-100">File Type</h3>
            <div className="flex flex-wrap gap-2">
              {['video/mp4', 'video/webm', 'video/mov'].map((type) => (
                <button
                  key={type}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    filterOptions.type.includes(type)
                      ? 'bg-primary text-white'
                      : 'bg-neutral-300 text-neutral-100 hover:bg-neutral-200'
                  }`}
                  onClick={() => {
                    const newTypes = filterOptions.type.includes(type)
                      ? filterOptions.type.filter(t => t !== type)
                      : [...filterOptions.type, type];
                    
                    setFilterOptions({ type: newTypes });
                  }}
                >
                  <FiFileText size={12} className="inline mr-1" />
                  {type.split('/')[1].toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <h3 className="font-medium text-sm mb-2 text-neutral-100">Tags</h3>
            <div className="flex flex-wrap gap-2">
              {['interview', 'marketing', 'tutorial', 'product', 'demo'].map((tag) => (
                <button
                  key={tag}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    filterOptions.tags.includes(tag)
                      ? 'bg-primary text-white'
                      : 'bg-neutral-300 text-neutral-100 hover:bg-neutral-200'
                  }`}
                  onClick={() => {
                    const newTags = filterOptions.tags.includes(tag)
                      ? filterOptions.tags.filter(t => t !== tag)
                      : [...filterOptions.tags, tag];
                    
                    setFilterOptions({ tags: newTags });
                  }}
                >
                  {tag.charAt(0).toUpperCase() + tag.slice(1)}
                </button>
              ))}
            </div>
          </div>
          
          <div className="pt-2 flex justify-end">
            <button
              className="text-sm text-primary hover:underline"
              onClick={() => setFilterOptions({
                status: ['processing', 'ready', 'error'],
                type: [],
                tags: []
              })}
            >
              Reset Filters
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
  
  return (
    <div className="bg-neutral-400 rounded-lg p-4 shadow-elevation-1">
      {/* Header & Tools */}
      <div className="flex flex-col md:flex-row md:justify-between md:items-center space-y-4 md:space-y-0 mb-6">
        <h2 className="text-xl font-bold text-neutral-100">Media Library</h2>
        
        <div className="flex flex-wrap items-center gap-2">
          {/* Search */}
          <div className="relative">
            <input
              type="text"
              placeholder="Search files..."
              className="pl-9 pr-4 py-2 bg-neutral-300 border border-neutral-300 rounded-md text-neutral-100 placeholder-neutral-200 focus:outline-none focus:ring-2 focus:ring-primary w-full md:w-64"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-200" size={16} />
            {searchQuery && (
              <button 
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-neutral-200 hover:text-neutral-100"
                onClick={() => setSearchQuery('')}
              >
                <FiX size={16} />
              </button>
            )}
          </div>
          
          {/* Filters */}
          <button
            className={`p-2 rounded-md transition-colors flex items-center space-x-1 ${
              showFilters ? 'bg-primary text-white' : 'bg-neutral-300 text-neutral-100 hover:bg-primary hover:text-white'
            }`}
            onClick={() => setShowFilters(!showFilters)}
          >
            <FiFilter size={16} />
            <span className="text-sm">Filter</span>
            <FiChevronDown 
              size={14} 
              className={`transform transition-transform ${showFilters ? 'rotate-180' : ''}`} 
            />
          </button>
          
          {/* Sort */}
          {renderSortOptions()}
          
          {/* View Toggle */}
          <div className="bg-neutral-300 rounded-md p-1 flex">
            <button
              className={`p-1.5 rounded ${viewMode === 'grid' ? 'bg-primary text-white' : 'text-neutral-100 hover:bg-neutral-200'}`}
              onClick={() => setViewMode('grid')}
              title="Grid View"
            >
              <FiGrid size={16} />
            </button>
            <button
              className={`p-1.5 rounded ${viewMode === 'list' ? 'bg-primary text-white' : 'text-neutral-100 hover:bg-neutral-200'}`}
              onClick={() => setViewMode('list')}
              title="List View"
            >
              <FiList size={16} />
            </button>
          </div>
        </div>
      </div>
      
      {/* Filter panel */}
      {renderFilterOptions()}
      
      {/* Selection Actions */}
      {selectedFiles.length > 0 && (
        <div className="bg-neutral-300 rounded-lg p-3 mb-4 flex justify-between items-center">
          <div className="text-sm text-neutral-100">
            <span className="font-medium">{selectedFiles.length}</span> file(s) selected
          </div>
          <div className="flex space-x-2">
            <button
              className="px-3 py-1.5 rounded-md text-sm font-medium bg-primary text-white hover:bg-primary-dark transition-colors flex items-center"
              onClick={() => {
                // In a real app, this would trigger the editor with the files
                alert('Edit selected files');
              }}
            >
              <FiEdit2 size={14} className="mr-1.5" />
              Edit
            </button>
            <button
              className="px-3 py-1.5 rounded-md text-sm font-medium bg-red-500 text-white hover:bg-red-600 transition-colors flex items-center"
              onClick={handleDeleteSelected}
            >
              <FiTrash2 size={14} className="mr-1.5" />
              Delete
            </button>
            <button
              className="px-3 py-1.5 rounded-md text-sm font-medium bg-neutral-400 text-neutral-100 hover:bg-neutral-500 transition-colors"
              onClick={deselectAllFiles}
            >
              Clear Selection
            </button>
          </div>
        </div>
      )}
      
      {/* Content */}
      <div className="relative min-h-[300px]">
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
          </div>
        ) : error ? (
          <div className="bg-red-500 bg-opacity-10 text-red-500 p-4 rounded-lg flex items-center">
            <FiAlertCircle size={20} className="mr-2" />
            <span>{error}</span>
          </div>
        ) : mediaFiles.length === 0 ? (
          <div className="bg-neutral-300 p-8 rounded-lg text-center">
            <p className="text-lg text-neutral-100 mb-4">No media files found</p>
            <p className="text-sm text-neutral-200 max-w-md mx-auto">
              Upload videos to get started with editing and processing. You can upload multiple files at once.
            </p>
          </div>
        ) : viewMode === 'grid' ? (
          <MediaGrid 
            files={mediaFiles} 
            onSelect={handleSelectFile} 
            onDetails={setDetailsFile} 
          />
        ) : (
          <MediaList 
            files={mediaFiles} 
            onSelect={handleSelectFile} 
            onDetails={setDetailsFile} 
          />
        )}
      </div>
      
      {/* File Details Modal */}
      {detailsFile && (
        <MediaDetails 
          file={detailsFile} 
          onClose={handleDetailsClose} 
        />
      )}
    </div>
  );
} 