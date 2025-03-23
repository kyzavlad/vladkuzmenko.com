'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiHome, 
  FiVideo, 
  FiUsers, 
  FiGlobe, 
  FiScissors, 
  FiFolder,
  FiGrid,
  FiSettings,
  FiHelpCircle,
  FiX,
  FiChevronRight,
  FiChevronDown,
} from 'react-icons/fi';

interface SidebarLinkProps {
  href: string;
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick?: () => void;
}

interface SidebarSubmenuProps {
  icon: React.ReactNode;
  label: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

interface DesktopSidebarProps {
  open: boolean;
  onClose: () => void;
}

const SidebarLink = ({ href, icon, label, active, onClick }: SidebarLinkProps) => {
  return (
    <Link 
      href={href}
      className={`flex items-center px-4 py-3 my-1 rounded-lg text-sm font-medium transition-colors ${
        active 
          ? 'bg-primary text-white' 
          : 'text-neutral-100 hover:bg-neutral-400 hover:text-white'
      }`}
      onClick={onClick}
    >
      <span className="mr-3">{icon}</span>
      <span>{label}</span>
    </Link>
  );
};

const SidebarSubmenu = ({ icon, label, children, defaultOpen = false }: SidebarSubmenuProps) => {
  const [isOpen, setIsOpen] = React.useState(defaultOpen);
  
  return (
    <div className="my-1">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full px-4 py-3 text-sm font-medium text-neutral-100 hover:bg-neutral-400 hover:text-white rounded-lg transition-colors"
      >
        <div className="flex items-center">
          <span className="mr-3">{icon}</span>
          <span>{label}</span>
        </div>
        <span className="transition-transform duration-200" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>
          <FiChevronDown size={16} />
        </span>
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden pl-3"
          >
            <div className="border-l-2 border-neutral-400 pl-4 py-1 ml-4">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default function DesktopSidebar({ open, onClose }: DesktopSidebarProps) {
  const pathname = usePathname();
  
  const isActive = (path: string) => {
    if (path === '/platform' || path === '/platform/dashboard') {
      return pathname === path || pathname === '/platform';
    }
    return pathname?.startsWith(path);
  };
  
  // Sidebar overlay for mobile
  const sidebarOverlay = (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 bg-black bg-opacity-50 z-20 md:hidden"
          onClick={onClose}
        />
      )}
    </AnimatePresence>
  );
  
  // Sidebar content
  const sidebarContent = (
    <div className="w-64 h-full flex flex-col">
      {/* Sidebar Header */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-neutral-400">
        <Link href="/platform" className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-primary rounded-md flex items-center justify-center">
            <span className="text-white font-bold">VP</span>
          </div>
          <span className="text-lg font-bold text-white">Video Platform</span>
        </Link>
        <button 
          className="md:hidden p-2 rounded-md text-neutral-200 hover:bg-neutral-400 hover:text-white"
          onClick={onClose}
        >
          <FiX size={20} />
        </button>
      </div>
      
      {/* Sidebar Navigation */}
      <div className="flex-1 overflow-y-auto py-4 px-3">
        <nav className="space-y-1">
          <SidebarLink 
            href="/platform/dashboard" 
            icon={<FiHome size={20} />} 
            label="Dashboard" 
            active={isActive('/platform/dashboard')}
          />
          
          <SidebarSubmenu 
            icon={<FiVideo size={20} />} 
            label="Video Tools" 
            defaultOpen={pathname?.includes('/platform/video-')}
          >
            <SidebarLink 
              href="/platform/video-editor" 
              icon={<FiVideo size={18} />} 
              label="Video Editor" 
              active={isActive('/platform/video-editor')}
            />
            <SidebarLink 
              href="/platform/video-templates" 
              icon={<FiGrid size={18} />} 
              label="Templates" 
              active={isActive('/platform/video-templates')}
            />
          </SidebarSubmenu>
          
          <SidebarLink 
            href="/platform/avatar-creator" 
            icon={<FiUsers size={20} />} 
            label="AI Avatars" 
            active={isActive('/platform/avatar-creator')}
          />
          
          <SidebarLink 
            href="/platform/translation" 
            icon={<FiGlobe size={20} />} 
            label="Translation" 
            active={isActive('/platform/translation')}
          />
          
          <SidebarLink 
            href="/platform/clip-generator" 
            icon={<FiScissors size={20} />} 
            label="Clip Generator" 
            active={isActive('/platform/clip-generator')}
          />
          
          <div className="pt-4 pb-2">
            <hr className="border-neutral-400" />
          </div>
          
          <SidebarLink 
            href="/platform/projects" 
            icon={<FiFolder size={20} />} 
            label="Projects" 
            active={isActive('/platform/projects')}
          />
          
          <SidebarLink 
            href="/platform/settings" 
            icon={<FiSettings size={20} />} 
            label="Settings" 
            active={isActive('/platform/settings')}
          />
          
          <SidebarLink 
            href="/platform/help" 
            icon={<FiHelpCircle size={20} />} 
            label="Help & Support" 
            active={isActive('/platform/help')}
          />
        </nav>
      </div>
      
      {/* Sidebar Footer */}
      <div className="border-t border-neutral-400 p-4">
        <div className="bg-neutral-400 rounded-lg p-3">
          <h4 className="text-sm font-medium text-white mb-2">Need more power?</h4>
          <p className="text-xs text-neutral-200 mb-3">Upgrade to Pro for additional processing credits and storage.</p>
          <Link
            href="/platform/billing/upgrade"
            className="block w-full py-2 px-3 bg-primary text-white text-center text-sm font-medium rounded-md hover:bg-primary-dark transition-colors"
          >
            Upgrade to Pro
          </Link>
        </div>
      </div>
    </div>
  );
  
  return (
    <>
      {/* Overlay */}
      {sidebarOverlay}
      
      {/* Sidebar container */}
      <div 
        className={`fixed md:static inset-y-0 left-0 z-30 transform transition-transform duration-300 ease-in-out bg-neutral-500 border-r border-neutral-400 md:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {sidebarContent}
      </div>
    </>
  );
} 