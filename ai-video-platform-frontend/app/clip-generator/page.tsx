'use client';

import React from 'react';
import ClipGeneratorHub from '../../components/clip-generator/clip-generator-hub';
import { ClipProvider } from '../../components/clip-generator/contexts/clip-context';

export default function ClipGeneratorPage() {
  return (
    <ClipProvider>
      <main className="container mx-auto px-4 py-8">
        <ClipGeneratorHub />
      </main>
    </ClipProvider>
  );
} 