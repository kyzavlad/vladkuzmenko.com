'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { FiVideo, FiUser, FiGlobe, FiBarChart2 } from 'react-icons/fi';

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 100,
      damping: 15
    }
  }
};

const DashboardCard = ({ 
  title, 
  description, 
  icon, 
  href, 
  color = 'bg-primary'
}: { 
  title: string; 
  description: string; 
  icon: React.ReactNode; 
  href: string;
  color?: string;
}) => {
  return (
    <motion.div variants={itemVariants}>
      <Link 
        href={href}
        className="block"
      >
        <div className={`card h-full cursor-pointer transform transition-all duration-300 hover:-translate-y-1 hover:shadow-elevation-3 border-l-4 ${color}`}>
          <div className="flex items-start space-x-4">
            <div className={`p-3 rounded-lg ${color.replace('bg-', 'bg-')} bg-opacity-20 text-${color.replace('bg-', '')}`}>
              {icon}
            </div>
            <div>
              <h3 className="text-lg font-medium">{title}</h3>
              <p className="text-neutral-200 mt-1">{description}</p>
            </div>
          </div>
        </div>
      </Link>
    </motion.div>
  );
};

const StatsCard = ({ title, value, change, icon }: { title: string; value: string; change?: string; icon: React.ReactNode }) => {
  const isPositive = change && !change.startsWith('-');
  
  return (
    <motion.div variants={itemVariants} className="card">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-neutral-200 text-sm">{title}</p>
          <h3 className="text-2xl font-semibold mt-1">{value}</h3>
          {change && (
            <p className={`text-sm ${isPositive ? 'text-green-500' : 'text-pink'} mt-1`}>
              {isPositive ? `+${change}` : change}
            </p>
          )}
        </div>
        <div className="p-3 rounded-lg bg-primary bg-opacity-20 text-primary">
          {icon}
        </div>
      </div>
    </motion.div>
  );
};

const RecentJobsTable = () => {
  const jobs = [
    { id: '1', name: 'Summer Promo Video', type: 'Video Edit', status: 'Completed', date: '2023-12-01' },
    { id: '2', name: 'Product Launch', type: 'Avatar Creation', status: 'Processing', date: '2023-12-05' },
    { id: '3', name: 'Training Materials', type: 'Translation', status: 'Pending', date: '2023-12-10' },
    { id: '4', name: 'Year in Review', type: 'Video Edit', status: 'Failed', date: '2023-12-12' },
  ];
  
  return (
    <motion.div variants={itemVariants} className="card overflow-hidden">
      <h3 className="text-lg font-medium mb-4">Recent Jobs</h3>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-neutral-300">
          <thead>
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-neutral-200 uppercase tracking-wider">Job Name</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-neutral-200 uppercase tracking-wider">Type</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-neutral-200 uppercase tracking-wider">Status</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-neutral-200 uppercase tracking-wider">Date</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-300">
            {jobs.map(job => (
              <tr key={job.id} className="hover:bg-neutral-300 transition-colors">
                <td className="px-4 py-3 whitespace-nowrap">{job.name}</td>
                <td className="px-4 py-3 whitespace-nowrap">{job.type}</td>
                <td className="px-4 py-3 whitespace-nowrap">
                  <span className={`inline-flex px-2 py-1 text-xs rounded-full ${
                    job.status === 'Completed' ? 'bg-green-100 text-green-800' :
                    job.status === 'Processing' ? 'bg-blue-100 text-blue-800' :
                    job.status === 'Pending' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {job.status}
                  </span>
                </td>
                <td className="px-4 py-3 whitespace-nowrap">{job.date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
};

export default function Dashboard() {
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-neutral-200 mt-1">Welcome to your AI Video Platform dashboard</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard 
          title="Total Videos" 
          value="42" 
          change="12%" 
          icon={<FiVideo size={24} />} 
        />
        <StatsCard 
          title="Total Avatars" 
          value="18" 
          change="5%" 
          icon={<FiUser size={24} />} 
        />
        <StatsCard 
          title="Total Translations" 
          value="36" 
          change="-3%" 
          icon={<FiGlobe size={24} />} 
        />
        <StatsCard 
          title="Processing Hours" 
          value="124" 
          change="8%" 
          icon={<FiBarChart2 size={24} />} 
        />
      </div>
      
      <h2 className="text-2xl font-bold mt-8">Quick Actions</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <DashboardCard
          title="Video Editor"
          description="Create and edit video clips with AI-powered tools"
          icon={<FiVideo size={24} />}
          href="/platform/video-editor"
          color="bg-primary"
        />
        <DashboardCard
          title="Avatar Creator"
          description="Generate custom AI avatars for your videos"
          icon={<FiUser size={24} />}
          href="/platform/avatar-creator"
          color="bg-purple"
        />
        <DashboardCard
          title="Translation Service"
          description="Translate your video content to multiple languages"
          icon={<FiGlobe size={24} />}
          href="/platform/translation"
          color="bg-pink"
        />
      </div>
      
      <RecentJobsTable />
    </motion.div>
  );
} 