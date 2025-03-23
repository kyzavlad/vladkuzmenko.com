import React from 'react';
import MainLayout from '../../components/layout/main-layout';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Platform | AI Video Platform',
  description: 'Manage your AI video projects, templates, and resources.'
};

export default function PlatformLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <MainLayout>
      {children}
    </MainLayout>
  );
} 