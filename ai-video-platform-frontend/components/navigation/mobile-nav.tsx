'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  FiHome,
  FiVideo,
  FiUsers,
  FiGlobe,
  FiPlus,
  FiX,
  FiFolder,
  FiChevronLeft,
  FiScissors
} from 'react-icons/fi';

interface BottomNavItemProps {
  icon: React.ReactNode;
  label: string;
  href: string;
  isActive: boolean;
}

const BottomNavItem = ({ icon, label, href, isActive }: BottomNavItemProps) => (
  <Link
    href={href}
    className="flex flex-col items-center justify-center px-2 py-1"
  >
    <div className={`p-1.5 rounded-full ${isActive ? 'bg-primary text-white' : 'text-neutral-100'}`}>
      {icon}
    </div>
    <span className={`text-xs mt-1 ${isActive ? 'text-primary' : 'text-neutral-200'}`}>{label}</span>
  </Link>
);

interface FloatingActionButtonProps {
  onClick: () => void;
}

const FloatingActionButton = ({ onClick }: FloatingActionButtonProps) => (
  <motion.button
    onClick={onClick}
    className="fixed bottom-20 right-4 w-14 h-14 rounded-full bg-primary text-white shadow-elevation-3 flex items-center justify-center"
    whileTap={{ scale: 0.95 }}
    whileHover={{ scale: 1.05 }}
  >
    <FiPlus size={24} />
  </motion.button>
);

interface QuickActionsMenuProps {
  isOpen: boolean;
  onClose: () => void;
}

const QuickActionsMenu = ({ isOpen, onClose }: QuickActionsMenuProps) => {
  const actions = [
    { label: 'New Video', icon: <FiVideo size={20} />, href: '/platform/video-editor/new' },
    { label: 'New Avatar', icon: <FiUsers size={20} />, href: '/platform/avatar-creator/new' },
    { label: 'New Translation', icon: <FiGlobe size={20} />, href: '/platform/translation/new' },
    { label: 'New Clip', icon: <FiScissors size={20} />, href: '/platform/clip-generator/new' },
  ];
  
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black bg-opacity-50 z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          
          {/* Menu */}
          <motion.div
            className="fixed bottom-24 right-4 bg-neutral-400 rounded-lg shadow-elevation-4 z-50 overflow-hidden"
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
          >
            <div className="p-2">
              {actions.map((action, index) => (
                <Link
                  key={index}
                  href={action.href}
                  className="flex items-center space-x-3 px-4 py-3 hover:bg-neutral-300 rounded-md transition-colors"
                  onClick={onClose}
                >
                  <div className="text-primary">{action.icon}</div>
                  <span className="text-neutral-100">{action.label}</span>
                </Link>
              ))}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default function MobileNav() {
  const pathname = usePathname();
  const [isQuickActionsOpen, setIsQuickActionsOpen] = useState(false);
  const [showBackButton, setShowBackButton] = useState(false);
  
  // Determine if we should show the back button
  // For example, if we're in a project or deep in a feature
  React.useEffect(() => {
    const pathParts = pathname?.split('/') || [];
    // Show back button if we're deeper than 2 levels in the path
    setShowBackButton(pathParts.length > 3);
  }, [pathname]);
  
  const navItems = [
    { icon: <FiHome size={20} />, label: 'Home', href: '/platform' },
    { icon: <FiVideo size={20} />, label: 'Videos', href: '/platform/video-editor' },
    { icon: <FiUsers size={20} />, label: 'Avatars', href: '/platform/avatar-creator' },
    { icon: <FiGlobe size={20} />, label: 'Translate', href: '/platform/translation' },
    { icon: <FiFolder size={20} />, label: 'Projects', href: '/platform/projects' },
  ];
  
  // Get the previous path for the back button
  const getPreviousPath = (): string => {
    if (!pathname) return '/platform';
    const parts = pathname.split('/');
    parts.pop();
    return parts.join('/') || '/platform';
  };

  return (
    <>
      {/* Contextual back button */}
      {showBackButton && (
        <div className="fixed top-16 left-0 z-30 p-2">
          <Link
            href={getPreviousPath()}
            className="bg-neutral-400 bg-opacity-80 backdrop-blur-sm rounded-full p-2 shadow-elevation-2 flex items-center"
          >
            <FiChevronLeft size={24} className="text-neutral-100" />
            <span className="ml-1 text-neutral-100 text-sm">Back</span>
          </Link>
        </div>
      )}
      
      {/* Floating action button */}
      <FloatingActionButton onClick={() => setIsQuickActionsOpen(true)} />
      
      {/* Quick actions menu */}
      <QuickActionsMenu 
        isOpen={isQuickActionsOpen} 
        onClose={() => setIsQuickActionsOpen(false)} 
      />
      
      {/* Bottom navigation bar */}
      <div className="md:hidden fixed bottom-0 w-full bg-neutral-400 shadow-elevation-3 border-t border-neutral-300 z-30">
        <div className="flex justify-around items-center h-16">
          {navItems.map((item) => (
            <BottomNavItem
              key={item.href}
              icon={item.icon}
              label={item.label}
              href={item.href}
              isActive={pathname === item.href}
            />
          ))}
        </div>
      </div>
    </>
  );
} 