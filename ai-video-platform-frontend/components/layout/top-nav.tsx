import React from 'react';
import { FiMenu, FiBell, FiUser } from 'react-icons/fi';

interface TopNavProps {
  onMenuButtonClick: () => void;
  sidebarOpen: boolean;
}

export default function TopNav({ onMenuButtonClick, sidebarOpen }: TopNavProps) {
  return (
    <nav className="bg-white dark:bg-neutral-800 border-b border-gray-200 dark:border-neutral-700">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <button
              onClick={onMenuButtonClick}
              className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-neutral-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary"
            >
              <span className="sr-only">Open sidebar</span>
              <FiMenu className="h-6 w-6" />
            </button>
          </div>

          <div className="flex items-center">
            <button
              type="button"
              className="p-2 rounded-full text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-neutral-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary"
            >
              <span className="sr-only">View notifications</span>
              <FiBell className="h-6 w-6" />
            </button>

            <button
              type="button"
              className="ml-4 p-2 rounded-full text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-neutral-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary"
            >
              <span className="sr-only">Open user menu</span>
              <FiUser className="h-6 w-6" />
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
} 