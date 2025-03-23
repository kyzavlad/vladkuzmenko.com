'use client';

import React from 'react';
import DashboardLayout from '../../../components/dashboard/dashboard-layout';
import { useAuth } from '../../../lib/auth/useAuth';
import { redirect } from 'next/navigation';

export default function DashboardPage() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (!user) {
    redirect('/login');
  }

  return <DashboardLayout />;
} 