'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { usePathname } from 'next/navigation';
import Logo from '../ui/logo';
import { 
  FiBell, 
  FiSearch, 
  FiUser, 
  FiSettings, 
  FiMenu, 
  FiX,
  FiVideo,
  FiUsers,
  FiGlobe,
  FiScissors,
  FiLogOut
} from 'react-icons/fi';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';

const NavItem = ({ href, label, icon, active = false }: { href: string; label: string; icon: React.ReactNode; active?: boolean }) => (
  <Link 
    href={href}
    className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      active 
        ? 'bg-primary bg-opacity-10 text-primary' 
        : 'text-neutral-100 hover:bg-neutral-400 hover:text-white'
    }`}
  >
    {icon}
    <span>{label}</span>
  </Link>
);

const SearchBar = () => {
  const [expanded, setExpanded] = useState(false);
  const [query, setQuery] = useState('');

  return (
    <div className="relative">
      <div className="flex items-center">
        <motion.div 
          className="relative flex items-center"
          animate={{ width: expanded ? 300 : 40 }}
          transition={{ duration: 0.3 }}
        >
          <input
            type="text"
            placeholder="Search..."
            className={`bg-neutral-400 text-neutral-100 rounded-full pl-10 pr-4 py-2 w-full focus:outline-none focus:ring-2 focus:ring-primary ${
              expanded ? 'opacity-100' : 'opacity-0'
            }`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onBlur={() => !query && setExpanded(false)}
          />
          <button 
            className="absolute left-0 top-0 p-2 rounded-full text-neutral-100 hover:text-primary transition-colors"
            onClick={() => setExpanded(!expanded)}
          >
            <FiSearch size={20} />
          </button>
        </motion.div>
      </div>
      
      {expanded && query && (
        <div className="absolute top-full right-0 mt-2 w-80 bg-neutral-400 rounded-lg shadow-elevation-3 p-3 z-20">
          <div className="font-medium mb-2">Quick Filters:</div>
          <div className="flex flex-wrap gap-1 mb-3">
            <span className="px-2 py-1 bg-neutral-300 rounded-full text-xs cursor-pointer hover:bg-primary hover:text-white transition-colors">Videos</span>
            <span className="px-2 py-1 bg-neutral-300 rounded-full text-xs cursor-pointer hover:bg-primary hover:text-white transition-colors">Avatars</span>
            <span className="px-2 py-1 bg-neutral-300 rounded-full text-xs cursor-pointer hover:bg-primary hover:text-white transition-colors">Templates</span>
            <span className="px-2 py-1 bg-neutral-300 rounded-full text-xs cursor-pointer hover:bg-primary hover:text-white transition-colors">Recent</span>
          </div>
          <div className="text-sm text-neutral-200">Press Enter to search...</div>
        </div>
      )}
    </div>
  );
};

const NotificationCenter = () => {
  const [isOpen, setIsOpen] = useState(false);
  const notifications = [
    { id: 1, title: 'Video processing complete', description: 'Your project "Summer Promo" is ready', time: '5 min ago', read: false },
    { id: 2, title: 'New template available', description: 'Check out the new Product Showcase template', time: '2 hours ago', read: false },
    { id: 3, title: 'System maintenance', description: 'Scheduled maintenance in 24 hours', time: '1 day ago', read: true },
  ];
  
  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <div className="relative">
      <button 
        className="p-2 rounded-full text-neutral-100 hover:bg-neutral-400 transition-colors relative"
        onClick={() => setIsOpen(!isOpen)}
      >
        <FiBell size={20} />
        {unreadCount > 0 && (
          <span className="absolute top-0 right-0 inline-flex items-center justify-center h-5 w-5 rounded-full bg-pink text-white text-xs">
            {unreadCount}
          </span>
        )}
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div 
            className="absolute top-full right-0 mt-2 w-80 bg-neutral-400 rounded-lg shadow-elevation-3 overflow-hidden z-20"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <div className="p-3 border-b border-neutral-300 flex justify-between items-center">
              <h3 className="font-medium">Notifications</h3>
              <button className="text-xs text-primary hover:underline">Mark all as read</button>
            </div>
            <div className="max-h-96 overflow-y-auto">
              {notifications.map(notification => (
                <div 
                  key={notification.id}
                  className={`p-3 border-b border-neutral-300 hover:bg-neutral-300 transition-colors cursor-pointer ${
                    !notification.read ? 'bg-neutral-300 bg-opacity-30' : ''
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <h4 className="font-medium text-sm">{notification.title}</h4>
                    <span className="text-xs text-neutral-200">{notification.time}</span>
                  </div>
                  <p className="text-sm text-neutral-200 mt-1">{notification.description}</p>
                </div>
              ))}
            </div>
            <div className="p-2 text-center">
              <Link href="/platform/notifications" className="text-sm text-primary hover:underline">
                View all notifications
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const UserMenu = () => {
  const [isOpen, setIsOpen] = useState(false);
  const user = useSelector((state: RootState) => state.auth.user);
  
  const userMenuItems = [
    { label: 'Profile', icon: <FiUser size={16} />, href: '/platform/profile' },
    { label: 'Settings', icon: <FiSettings size={16} />, href: '/platform/settings' },
    { label: 'Logout', icon: <FiLogOut size={16} />, href: '/logout' },
  ];

  return (
    <div className="relative">
      <button 
        className="flex items-center space-x-2 p-2 rounded-full hover:bg-neutral-400 transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white overflow-hidden">
          {user?.avatar ? (
            <img src={user.avatar} alt={user.name} className="w-full h-full object-cover" />
          ) : (
            <FiUser size={20} />
          )}
        </div>
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div 
            className="absolute top-full right-0 mt-2 w-56 bg-neutral-400 rounded-lg shadow-elevation-3 overflow-hidden z-20"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <div className="p-3 border-b border-neutral-300">
              <div className="font-medium">{user?.name || 'User'}</div>
              <div className="text-sm text-neutral-200">{user?.email || 'user@example.com'}</div>
            </div>
            <div className="py-1">
              {userMenuItems.map((item, index) => (
                <Link 
                  key={index}
                  href={item.href}
                  className="flex items-center space-x-2 px-4 py-2 text-sm text-neutral-100 hover:bg-neutral-300 transition-colors"
                  onClick={() => setIsOpen(false)}
                >
                  <span className="text-neutral-200">{item.icon}</span>
                  <span>{item.label}</span>
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default function TopNav() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  const navItems = [
    { href: '/platform/video-editor', label: 'Video Editor', icon: <FiVideo size={16} className="mr-1" /> },
    { href: '/platform/avatar-creator', label: 'Avatar Creator', icon: <FiUsers size={16} className="mr-1" /> },
    { href: '/platform/translation', label: 'Translation', icon: <FiGlobe size={16} className="mr-1" /> },
    { href: '/platform/clip-generator', label: 'Clip Generator', icon: <FiScissors size={16} className="mr-1" /> },
  ];

  return (
    <header className="sticky top-0 bg-neutral-500 z-10 border-b border-neutral-400 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <Logo />
          </div>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center justify-center flex-1 px-8">
            <nav className="flex space-x-2">
              {navItems.map((item) => (
                <NavItem 
                  key={item.href}
                  href={item.href}
                  label={item.label}
                  icon={item.icon}
                  active={pathname === item.href}
                />
              ))}
            </nav>
          </div>
          
          {/* Right side items */}
          <div className="hidden md:flex items-center space-x-2">
            <SearchBar />
            <NotificationCenter />
            <UserMenu />
          </div>
          
          {/* Mobile menu button */}
          <div className="md:hidden flex items-center space-x-2">
            <SearchBar />
            <NotificationCenter />
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-full text-neutral-100 hover:bg-neutral-400 transition-colors"
            >
              {mobileMenuOpen ? <FiX size={24} /> : <FiMenu size={24} />}
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            className="md:hidden bg-neutral-500 shadow-lg"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="px-4 pt-2 pb-4 space-y-1 border-t border-neutral-400">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center px-3 py-2 rounded-md text-base font-medium ${
                    pathname === item.href
                      ? 'bg-primary bg-opacity-10 text-primary'
                      : 'text-neutral-100 hover:bg-neutral-400 hover:text-white'
                  }`}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  {item.icon}
                  <span>{item.label}</span>
                </Link>
              ))}
              <div className="pt-2 mt-2 border-t border-neutral-400">
                <UserMenu />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
} 