'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { FiCpu, FiInfo, FiHardDrive, FiClock, FiArrowUpRight } from 'react-icons/fi';
import Link from 'next/link';

interface UsageMetric {
  id: string;
  title: string;
  icon: React.ReactNode;
  value: string | number;
  max?: string | number;
  percentage?: number;
  color: string;
  link: string;
}

export default function ResourceUsage() {
  const usageMetrics: UsageMetric[] = [
    {
      id: 'processing-credits',
      title: 'Processing Credits',
      icon: <FiCpu size={20} />,
      value: 850,
      max: 1000,
      percentage: 85,
      color: 'primary',
      link: '/platform/billing/credits'
    },
    {
      id: 'storage',
      title: 'Storage Used',
      icon: <FiHardDrive size={20} />,
      value: '2.8 GB',
      max: '10 GB',
      percentage: 28,
      color: 'teal',
      link: '/platform/storage'
    },
    {
      id: 'render-time',
      title: 'Render Time',
      icon: <FiClock size={20} />,
      value: '5.3h',
      max: '10h',
      percentage: 53,
      color: 'purple',
      link: '/platform/usage/render-time'
    }
  ];

  return (
    <div className="mb-8">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Resource Usage</h2>
        <Link 
          href="/platform/usage"
          className="text-primary text-sm hover:underline flex items-center"
        >
          View Usage Details
          <FiArrowUpRight size={16} className="ml-1" />
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {usageMetrics.map((metric) => (
          <UsageCard key={metric.id} metric={metric} />
        ))}
      </div>
    </div>
  );
}

const UsageCard = ({ metric }: { metric: UsageMetric }) => {
  // Get color classes based on metric.color
  const getColorClass = (type: 'bg' | 'text' | 'border') => {
    const baseClass = `${type}-${metric.color}`;
    if (type === 'bg') return `${baseClass} ${baseClass}-opacity-10`;
    return baseClass;
  };

  // Get percentage color based on value
  const getPercentageColor = (percentage: number) => {
    if (percentage >= 90) return 'text-red-500';
    if (percentage >= 75) return 'text-orange-500';
    if (percentage >= 50) return 'text-yellow-500';
    return 'text-green-500';
  };

  return (
    <motion.div
      className="bg-neutral-400 rounded-lg p-4 shadow-elevation-1 hover:shadow-elevation-2 transition-shadow"
      whileHover={{ y: -3 }}
      transition={{ duration: 0.2 }}
    >
      <Link href={metric.link} className="block">
        <div className="flex items-center space-x-3 mb-3">
          <div className={`p-2 rounded-md ${getColorClass('bg')} ${getColorClass('text')}`}>
            {metric.icon}
          </div>
          <h3 className="font-medium text-neutral-100">
            {metric.title}
          </h3>
          <button 
            className="text-neutral-200 hover:text-neutral-100 transition-colors ml-auto"
            title="More information"
          >
            <FiInfo size={16} />
          </button>
        </div>

        <div className="flex items-baseline space-x-2 mb-2">
          <span className="text-2xl font-bold">{metric.value}</span>
          {metric.max && (
            <span className="text-neutral-200">
              / {metric.max}
            </span>
          )}
          {metric.percentage !== undefined && (
            <span className={`ml-auto ${getPercentageColor(metric.percentage)}`}>
              {metric.percentage}%
            </span>
          )}
        </div>

        {metric.percentage !== undefined && (
          <div className="w-full bg-neutral-300 rounded-full h-2 overflow-hidden">
            <div 
              className={`h-full ${getColorClass('bg')}`}
              style={{ width: `${metric.percentage}%` }}
            ></div>
          </div>
        )}
      </Link>
    </motion.div>
  );
}; 