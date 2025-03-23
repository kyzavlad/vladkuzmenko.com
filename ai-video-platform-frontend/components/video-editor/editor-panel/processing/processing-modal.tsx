'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { FiX, FiCpu, FiHardDrive, FiZap } from 'react-icons/fi';
import { useProcessingContext } from '../../contexts/processing-context';

interface ProcessingModalProps {
  onClose: () => void;
}

const ProcessingModal: React.FC<ProcessingModalProps> = ({ onClose }) => {
  const { activeJob, isProcessing } = useProcessingContext();

  if (!activeJob || !isProcessing) return null;

  const formatTime = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatPercentage = (value: number): string => {
    return `${Math.round(value)}%`;
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4"
      >
        <div className="p-6">
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-xl font-semibold">Processing Video</h2>
              <p className="text-gray-500 mt-1">Please wait while we process your video</p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <FiX size={24} />
            </button>
          </div>

          {/* Overall Progress */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">Overall Progress</span>
              <span className="text-sm text-gray-500">
                {formatPercentage(activeJob.progress)}
              </span>
            </div>
            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-300"
                style={{ width: `${activeJob.progress}%` }}
              />
            </div>
          </div>

          {/* Processing Stages */}
          <div className="space-y-4">
            {activeJob.stages.map((stage, index) => (
              <div key={index}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">{stage.name}</span>
                  <span className="text-sm text-gray-500">
                    {formatPercentage(stage.progress)}
                  </span>
                </div>
                <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-300"
                    style={{ width: `${stage.progress}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Resource Usage */}
          {activeJob.resourceUsage && (
            <div className="mt-6 grid grid-cols-3 gap-4">
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-2">
                  <FiCpu className="text-gray-400" />
                  <span className="text-sm font-medium">CPU</span>
                </div>
                <div className="text-lg font-semibold">
                  {formatPercentage(activeJob.resourceUsage.cpu)}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-2">
                  <FiHardDrive className="text-gray-400" />
                  <span className="text-sm font-medium">Memory</span>
                </div>
                <div className="text-lg font-semibold">
                  {formatPercentage(activeJob.resourceUsage.memory)}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-2">
                  <FiZap className="text-gray-400" />
                  <span className="text-sm font-medium">GPU</span>
                </div>
                <div className="text-lg font-semibold">
                  {formatPercentage(activeJob.resourceUsage.gpu)}
                </div>
              </div>
            </div>
          )}

          {/* Estimated Time */}
          {activeJob.estimatedTimeRemaining !== undefined && (
            <div className="mt-6 text-center text-sm text-gray-500">
              Estimated time remaining:{' '}
              <span className="font-medium">
                {formatTime(activeJob.estimatedTimeRemaining)}
              </span>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default ProcessingModal; 