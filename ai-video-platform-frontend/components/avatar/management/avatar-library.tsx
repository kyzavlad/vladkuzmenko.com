'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FiPlus, 
  FiTrash2, 
  FiEdit3, 
  FiPlay, 
  FiDownload, 
  FiShare2, 
  FiMoreVertical,
  FiClock,
  FiVideo,
  FiStar,
  FiAlertCircle
} from 'react-icons/fi';
import { useAvatarContext, Avatar } from '../contexts/avatar-context';
import Link from 'next/link';
import { formatDate } from '../../../utils/date';

export const AvatarLibrary: React.FC = () => {
  const { avatars, fetchAvatars, isLoading, error } = useAvatarContext();
  const [sortBy, setSortBy] = useState<'name' | 'created'>('created');
  const [filterBy, setFilterBy] = useState<string>('all');

  useEffect(() => {
    fetchAvatars();
  }, [fetchAvatars]);

  const sortAvatars = (avatars: Avatar[]) => {
    return [...avatars].sort((a, b) => {
      if (sortBy === 'name') {
        return a.name.localeCompare(b.name);
      }
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    }).filter(avatar => {
      if (filterBy === 'all') return true;
      return avatar.style.id === filterBy;
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-red-500">{error}</div>
      </div>
    );
  }

  const sortedAvatars = sortAvatars(avatars);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Avatar Library</h2>
        <div className="flex gap-4">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'name' | 'created')}
            className="px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="name">Sort by Name</option>
            <option value="created">Sort by Created Date</option>
          </select>
          <select
            value={filterBy}
            onChange={(e) => setFilterBy(e.target.value)}
            className="px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="all">All Styles</option>
            {Array.from(new Set(avatars.map(avatar => avatar.style.id))).map(styleId => (
              <option key={styleId} value={styleId}>
                {styleId}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {sortedAvatars.map((avatar) => (
          <motion.div
            key={avatar.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-lg overflow-hidden"
          >
            <div className="aspect-square rounded-lg overflow-hidden">
              <img 
                src={avatar.thumbnail} 
                alt={avatar.name}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="p-4">
              <h3 className="text-lg font-semibold">{avatar.name}</h3>
              <p className="text-sm text-gray-500">
                Created {formatDate(new Date(avatar.createdAt))}
              </p>
              <div className="mt-2 flex items-center gap-2">
                <span className="text-sm text-gray-500">
                  Style: {avatar.style.id}
                </span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}; 