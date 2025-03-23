'use client';

import React, { useState } from 'react';
import { 
  FiVideo, 
  FiEdit3, 
  FiTrash2, 
  FiShare2, 
  FiDownload,
  FiChevronRight,
  FiUpload
} from 'react-icons/fi';
import { formatRelativeTime } from '../../lib/utils/formatters';

type ActivityType = 
  | 'all'
  | 'create'
  | 'edit'
  | 'share'
  | 'export'
  | 'import'
  | 'complete'
  | 'delete';

interface Activity {
  id: string;
  type: ActivityType;
  title: string;
  description: string;
  timestamp: string;
  status?: 'pending' | 'completed' | 'failed';
  progress?: number;
}

export default function ActivityFeed() {
  const [filter, setFilter] = useState<ActivityType>('all');
  const activities: Activity[] = []; // В реальном приложении здесь будут данные

  const filterTabs: ActivityType[] = ['all', 'create', 'edit', 'share', 'export', 'import', 'complete', 'delete'];

  return (
    <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-neutral-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Activity Feed</h2>
      </div>

      {/* Filter tabs */}
      <div className="flex space-x-1 mb-4 overflow-x-auto pb-1 hide-scrollbar">
        {filterTabs.map((type) => (
          <button
            key={type}
            className={`
              px-3 py-1.5 rounded-lg text-sm whitespace-nowrap
              ${filter === type
                ? 'bg-primary text-white'
                : 'bg-neutral-300 text-neutral-200 hover:bg-neutral-200 hover:text-neutral-100'
              } transition-colors
            `}
            onClick={() => setFilter(type)}
          >
            {type === 'all' ? 'All Activities' : `${type.charAt(0).toUpperCase()}${type.slice(1)}`}
          </button>
        ))}
      </div>

      {/* Activity list */}
      <div className="divide-y divide-gray-200 dark:divide-neutral-700">
        {activities
          .filter(activity => filter === 'all' || activity.type === filter)
          .map(activity => (
            <div key={activity.id} className="p-4 hover:bg-gray-50 dark:hover:bg-neutral-700/50 transition-colors">
              <div className="flex items-start">
                {/* Icon */}
                <div className="flex-shrink-0 mt-1">
                  {activity.type === 'create' && <FiVideo className="text-blue-500" size={18} />}
                  {activity.type === 'edit' && <FiEdit3 className="text-yellow-500" size={18} />}
                  {activity.type === 'share' && <FiShare2 className="text-green-500" size={18} />}
                  {activity.type === 'export' && <FiDownload className="text-purple-500" size={18} />}
                  {activity.type === 'import' && <FiUpload className="text-orange-500" size={18} />}
                  {activity.type === 'delete' && <FiTrash2 className="text-red-500" size={18} />}
                </div>

                {/* Content */}
                <div className="ml-3 flex-1">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-gray-900 dark:text-white">
                      {activity.title}
                    </p>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {formatRelativeTime(activity.timestamp)}
                    </span>
                  </div>
                  
                  <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    {activity.description}
                  </p>

                  {/* Status and progress */}
                  {activity.status && (
                    <div className="mt-2">
                      <div className="flex items-center">
                        <span className={`
                          text-xs font-medium mr-2
                          ${activity.status === 'completed' ? 'text-green-500' : ''}
                          ${activity.status === 'pending' ? 'text-yellow-500' : ''}
                          ${activity.status === 'failed' ? 'text-red-500' : ''}
                        `}>
                          {activity.status.charAt(0).toUpperCase() + activity.status.slice(1)}
                        </span>
                        
                        {activity.progress !== undefined && activity.status === 'pending' && (
                          <div className="flex-1 h-1 bg-gray-200 dark:bg-neutral-600 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-primary rounded-full transition-all duration-500"
                              style={{ width: `${activity.progress}%` }}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Action button */}
                <button className="ml-4 p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                  <FiChevronRight size={16} />
                </button>
              </div>
            </div>
          ))}

        {/* Empty state */}
        {activities.length === 0 && (
          <div className="p-4 text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">No activities to show</p>
          </div>
        )}
      </div>
    </div>
  );
} 