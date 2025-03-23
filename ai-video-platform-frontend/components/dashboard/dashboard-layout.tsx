'use client';

import React from 'react';
import WelcomeBanner from './welcome-banner';
import QuickActions from './quick-actions';
import RecentProjects from './recent-projects';
import ResourceUsage from './resource-usage';
import ActivityFeed from './activity-feed';
import FeaturedTemplates from './featured-templates';

interface DashboardLayoutProps {
  showWelcomeBanner?: boolean;
}

export default function DashboardLayout({ 
  showWelcomeBanner = true 
}: DashboardLayoutProps) {
  
  return (
    <div className="p-4 md:p-6 max-w-screen-2xl mx-auto">
      <div className="grid grid-cols-1 gap-6">
        {/* Welcome Banner */}
        {showWelcomeBanner && (
          <div className="col-span-1">
            <WelcomeBanner />
          </div>
        )}
        
        {/* Quick Actions */}
        <div className="col-span-1">
          <QuickActions />
        </div>
        
        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column (2/3 width on large screens) */}
          <div className="lg:col-span-2 space-y-6">
            {/* Recent Projects */}
            <RecentProjects />
            
            {/* Featured Templates */}
            <FeaturedTemplates />
          </div>
          
          {/* Right Column (1/3 width on large screens) */}
          <div className="lg:col-span-1 space-y-6">
            {/* Resource Usage */}
            <ResourceUsage />
            
            {/* Activity Feed */}
            <ActivityFeed />
          </div>
        </div>
      </div>
    </div>
  );
} 