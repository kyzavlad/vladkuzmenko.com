'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  FiHome,
  FiVideo,
  FiUsers,
  FiGlobe,
  FiScissors,
  FiFolder,
  FiCode,
  FiClock,
  FiChevronLeft,
  FiChevronRight,
  FiGrid,
  FiPieChart,
  FiPlus,
  FiList
} from 'react-icons/fi';
import { truncateString } from '../../lib/utils/formatters';

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  href: string;
  isActive?: boolean;
  isCollapsed?: boolean;
  badge?: number | string;
}

const NavItem = ({ icon, label, href, isActive = false, isCollapsed = false, badge }: NavItemProps) => (
  <Link 
    href={href}
    className={`flex items-center px-3 py-2 rounded-md transition-colors ${
      isActive 
        ? 'bg-primary text-white' 
        : 'text-neutral-100 hover:bg-neutral-300'
    }`}
  >
    <div className="flex-shrink-0 mr-3">{icon}</div>
    {!isCollapsed && (
      <div className="flex-1">
        <span className={`${isActive ? 'font-medium' : ''}`}>{label}</span>
      </div>
    )}
    {badge && !isCollapsed && (
      <div className="ml-auto">
        <span className="bg-pink text-white text-xs px-2 py-1 rounded-full">{badge}</span>
      </div>
    )}
  </Link>
);

const ProjectItem = ({ name, type, thumbnail, href, isCollapsed = false }: { 
  name: string; 
  type: string; 
  thumbnail?: string; 
  href: string; 
  isCollapsed?: boolean;
}) => (
  <Link href={href} className="flex items-center px-3 py-2 rounded-md hover:bg-neutral-300 transition-colors">
    <div className="flex-shrink-0 w-8 h-8 mr-3 rounded bg-neutral-300 overflow-hidden">
      {thumbnail ? (
        <img src={thumbnail} alt={name} className="w-full h-full object-cover" />
      ) : (
        <div className="flex items-center justify-center w-full h-full text-neutral-100">
          {type === 'video' && <FiVideo />}
          {type === 'avatar' && <FiUsers />}
          {type === 'translation' && <FiGlobe />}
        </div>
      )}
    </div>
    {!isCollapsed && (
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate">{truncateString(name, 20)}</div>
        <div className="text-xs text-neutral-200">{type}</div>
      </div>
    )}
  </Link>
);

export default function SidebarNav() {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  // Main navigation items
  const navItems = [
    { icon: <FiHome size={20} />, label: 'Dashboard', href: '/platform' },
    { icon: <FiVideo size={20} />, label: 'Video Editor', href: '/platform/video-editor', badge: 'New' },
    { icon: <FiUsers size={20} />, label: 'Avatar Creator', href: '/platform/avatar-creator' },
    { icon: <FiGlobe size={20} />, label: 'Translation', href: '/platform/translation' },
    { icon: <FiScissors size={20} />, label: 'Clip Generator', href: '/platform/clip-generator' },
  ];
  
  // Recent projects
  const recentProjects = [
    { name: 'Summer Promotion', type: 'video', href: '/platform/video-editor/project/1' },
    { name: 'CEO Avatar', type: 'avatar', href: '/platform/avatar-creator/project/2' },
    { name: 'Product Demo', type: 'video', href: '/platform/video-editor/project/3' },
    { name: 'Help Documentation', type: 'translation', href: '/platform/translation/project/4' },
  ];
  
  // Template categories
  const templateCategories = [
    { name: 'Marketing', count: 12 },
    { name: 'Educational', count: 8 },
    { name: 'Social Media', count: 15 },
    { name: 'Presentations', count: 7 },
  ];
  
  // Usage statistics
  const usageStats = {
    tokensRemaining: 2500,
    tokensTotal: 5000,
    storageUsed: '2.1 GB',
    storageTotal: '5 GB',
  };

  const containerVariants = {
    expanded: { width: 260 },
    collapsed: { width: 70 },
  };

  return (
    <motion.div
      className="h-full bg-neutral-400 flex flex-col border-r border-neutral-300 overflow-hidden"
      initial={isCollapsed ? 'collapsed' : 'expanded'}
      animate={isCollapsed ? 'collapsed' : 'expanded'}
      variants={containerVariants}
      transition={{ duration: 0.3 }}
    >
      {/* Collapse toggle button */}
      <div className="flex justify-end p-2">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1 rounded-md text-neutral-200 hover:bg-neutral-300 hover:text-neutral-100"
        >
          {isCollapsed ? <FiChevronRight size={20} /> : <FiChevronLeft size={20} />}
        </button>
      </div>
      
      {/* Main navigation */}
      <div className="px-3 py-2">
        <nav className="space-y-1">
          {navItems.map((item) => (
            <NavItem
              key={item.href}
              icon={item.icon}
              label={item.label}
              href={item.href}
              isActive={pathname === item.href}
              isCollapsed={isCollapsed}
              badge={item.badge}
            />
          ))}
        </nav>
      </div>
      
      {/* Recent projects section */}
      {!isCollapsed && (
        <div className="px-3 py-4 border-t border-neutral-300 flex-grow overflow-auto">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-neutral-100">Recent Projects</h3>
            <Link href="/platform/projects" className="text-primary text-xs hover:underline">
              View All
            </Link>
          </div>
          <div className="space-y-2">
            {recentProjects.map((project, index) => (
              <ProjectItem
                key={index}
                name={project.name}
                type={project.type}
                href={project.href}
                isCollapsed={isCollapsed}
              />
            ))}
          </div>
        </div>
      )}
      
      {/* Template section */}
      {!isCollapsed && (
        <div className="px-3 py-4 border-t border-neutral-300">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-neutral-100">Templates</h3>
            <Link href="/platform/templates" className="text-primary text-xs hover:underline">
              Browse
            </Link>
          </div>
          <div className="space-y-1">
            {templateCategories.map((category, index) => (
              <Link
                key={index}
                href={`/platform/templates/${category.name.toLowerCase()}`}
                className="flex items-center justify-between px-3 py-1.5 text-sm rounded-md hover:bg-neutral-300 transition-colors"
              >
                <span className="text-neutral-100">{category.name}</span>
                <span className="text-xs text-neutral-200">{category.count}</span>
              </Link>
            ))}
          </div>
          <button className="mt-3 flex items-center justify-center w-full px-3 py-2 text-sm bg-primary bg-opacity-10 text-primary rounded-md hover:bg-opacity-20 transition-colors">
            <FiPlus size={16} className="mr-2" />
            <span>New from Template</span>
          </button>
        </div>
      )}
      
      {/* Usage statistics */}
      {!isCollapsed && (
        <div className="px-3 py-4 border-t border-neutral-300">
          <div className="mb-3">
            <h3 className="text-sm font-medium text-neutral-100 flex items-center">
              <FiPieChart size={16} className="mr-2" />
              <span>Usage Statistics</span>
            </h3>
          </div>
          
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-neutral-200">Tokens</span>
                <span className="text-neutral-100">{usageStats.tokensRemaining} / {usageStats.tokensTotal}</span>
              </div>
              <div className="w-full h-2 bg-neutral-300 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full" 
                  style={{ width: `${(usageStats.tokensRemaining / usageStats.tokensTotal) * 100}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-neutral-200">Storage</span>
                <span className="text-neutral-100">{usageStats.storageUsed} / {usageStats.storageTotal}</span>
              </div>
              <div className="w-full h-2 bg-neutral-300 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-purple rounded-full" 
                  style={{ width: '42%' }}
                />
              </div>
            </div>
          </div>
          
          <Link 
            href="/platform/billing" 
            className="mt-4 flex items-center justify-center w-full px-3 py-2 text-sm bg-neutral-300 text-neutral-100 rounded-md hover:bg-neutral-200 transition-colors"
          >
            Upgrade Plan
          </Link>
        </div>
      )}
      
      {/* Collapsed placeholder elements */}
      {isCollapsed && (
        <>
          <div className="flex-grow border-t border-neutral-300 px-3 py-4">
            <div className="flex justify-center">
              <FiFolder size={20} className="text-neutral-200" />
            </div>
          </div>
          <div className="border-t border-neutral-300 px-3 py-4">
            <div className="flex justify-center">
              <FiGrid size={20} className="text-neutral-200" />
            </div>
          </div>
          <div className="border-t border-neutral-300 px-3 py-4">
            <div className="flex justify-center">
              <FiPieChart size={20} className="text-neutral-200" />
            </div>
          </div>
        </>
      )}
    </motion.div>
  );
} 