'use client';

import React, { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import DesktopSidebar from '../navigation/desktop-sidebar';
import TopNav from './top-nav';
import MobileNav from '../navigation/mobile-nav';

interface MainLayoutProps {
  children: React.ReactNode;
}

export default function MainLayout({ children }: MainLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const pathname = usePathname();
  
  // Close sidebar when route changes
  useEffect(() => {
    setSidebarOpen(false);
  }, [pathname]);
  
  // Function to toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };
  
  return (
    <div className="flex h-screen bg-neutral-500">
      {/* Desktop Sidebar */}
      <DesktopSidebar 
        open={sidebarOpen} 
        onClose={() => setSidebarOpen(false)} 
      />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Navigation */}
        <TopNav 
          onMenuButtonClick={toggleSidebar} 
          sidebarOpen={sidebarOpen} 
        />
        
        {/* Page Content */}
        <main className="flex-1 overflow-y-auto bg-neutral-500">
          {children}
        </main>
        
        {/* Mobile Navigation */}
        <div className="md:hidden">
          <MobileNav />
        </div>
      </div>
    </div>
  );
} 