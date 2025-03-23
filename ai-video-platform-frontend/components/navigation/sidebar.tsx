'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  FiHome, FiUser, FiVideo, FiSettings, 
  FiFileText, FiMic, FiGlobe, FiScissors,
  FiPlusCircle, FiChevronDown, FiChevronRight, FiHelpCircle
} from 'react-icons/fi';

interface NavigationItem {
  name: string;
  path: string;
  icon: React.ReactNode;
  badge?: {
    text: string;
    color: string;
  };
  children?: Omit<NavigationItem, 'children'>[];
}

export const Sidebar: React.FC = () => {
  const pathname = usePathname();
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  
  const navigationItems: NavigationItem[] = [
    {
      name: 'Dashboard',
      path: '/',
      icon: <FiHome className="w-5 h-5" />
    },
    {
      name: 'Avatar Creation',
      path: '/avatar',
      icon: <FiUser className="w-5 h-5" />,
      badge: {
        text: 'New',
        color: 'bg-blue-500'
      },
      children: [
        {
          name: 'Create New Avatar',
          path: '/avatar/create',
          icon: <FiPlusCircle className="w-4 h-4" />
        },
        {
          name: 'My Avatars',
          path: '/avatar/library',
          icon: <FiUser className="w-4 h-4" />
        },
        {
          name: 'Generate Video',
          path: '/avatar/generate',
          icon: <FiVideo className="w-4 h-4" />
        }
      ]
    },
    {
      name: 'Video Translation',
      path: '/translation',
      icon: <FiGlobe className="w-5 h-5" />,
      children: [
        {
          name: 'New Translation',
          path: '/translation/new',
          icon: <FiPlusCircle className="w-4 h-4" />
        },
        {
          name: 'My Translations',
          path: '/translation/library',
          icon: <FiFileText className="w-4 h-4" />
        }
      ]
    },
    {
      name: 'Clip Generator',
      path: '/clip-generator',
      icon: <FiScissors className="w-5 h-5" />,
      badge: {
        text: 'Beta',
        color: 'bg-purple-500'
      },
      children: [
        {
          name: 'Generate Clips',
          path: '/clip-generator/upload',
          icon: <FiPlusCircle className="w-4 h-4" />
        },
        {
          name: 'My Clips',
          path: '/clip-generator/library',
          icon: <FiVideo className="w-4 h-4" />
        }
      ]
    },
    {
      name: 'Voice Cloning',
      path: '/voice-cloning',
      icon: <FiMic className="w-5 h-5" />
    },
    {
      name: 'Settings',
      path: '/settings',
      icon: <FiSettings className="w-5 h-5" />
    },
    {
      name: 'Help & Support',
      path: '/support',
      icon: <FiHelpCircle className="w-5 h-5" />
    }
  ];
  
  const toggleGroup = (name: string) => {
    setExpandedGroups(prev => ({
      ...prev,
      [name]: !prev[name]
    }));
  };
  
  const isActive = (path: string) => {
    return pathname === path || pathname?.startsWith(path + '/');
  };
  
  useEffect(() => {
    navigationItems.forEach(item => {
      if (item.children && item.children.some(child => isActive(child.path))) {
        setExpandedGroups(prev => ({
          ...prev,
          [item.name]: true
        }));
      }
    });
  }, [pathname]);
  
  return (
    <aside className="w-64 bg-white dark:bg-gray-800 h-screen overflow-y-auto shadow-md fixed left-0 top-0 z-10 transition-all">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <Link href="/" className="flex items-center">
          <img src="/logo.svg" alt="Logo" className="h-8 w-8" />
          <span className="ml-2 text-xl font-semibold text-gray-800 dark:text-white">AI Video Platform</span>
        </Link>
      </div>
      
      <nav className="mt-4 px-4">
        <ul className="space-y-1">
          {navigationItems.map((item) => (
            <li key={item.name}>
              {item.children ? (
                <div>
                  <button
                    onClick={() => toggleGroup(item.name.toLowerCase())}
                    className={`w-full flex items-center justify-between p-2 rounded-md ${
                      isActive(item.path) 
                        ? 'bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-200' 
                        : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center">
                      {item.icon}
                      <span className="ml-3">{item.name}</span>
                      {item.badge && (
                        <span className={`ml-2 text-xs px-1.5 py-0.5 rounded-full text-white ${item.badge.color}`}>
                          {item.badge.text}
                        </span>
                      )}
                    </div>
                    {expandedGroups[item.name.toLowerCase()] ? (
                      <FiChevronDown className="w-4 h-4" />
                    ) : (
                      <FiChevronRight className="w-4 h-4" />
                    )}
                  </button>
                  
                  {expandedGroups[item.name.toLowerCase()] && (
                    <ul className="pl-8 mt-1 space-y-1">
                      {item.children.map((child) => (
                        <li key={child.name}>
                          <Link
                            href={child.path}
                            className={`flex items-center p-2 rounded-md ${
                              isActive(child.path) 
                                ? 'bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-200' 
                                : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                            }`}
                          >
                            {child.icon}
                            <span className="ml-3 text-sm">{child.name}</span>
                            {child.badge && (
                              <span className={`ml-2 text-xs px-1.5 py-0.5 rounded-full text-white ${child.badge.color}`}>
                                {child.badge.text}
                              </span>
                            )}
                          </Link>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : (
                <Link 
                  href={item.path}
                  className={`flex items-center p-2 rounded-md ${
                    isActive(item.path) 
                      ? 'bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-200' 
                      : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                  }`}
                >
                  {item.icon}
                  <span className="ml-3">{item.name}</span>
                  {item.badge && (
                    <span className={`ml-2 text-xs px-1.5 py-0.5 rounded-full text-white ${item.badge.color}`}>
                      {item.badge.text}
                    </span>
                  )}
                </Link>
              )}
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
};

export default Sidebar; 