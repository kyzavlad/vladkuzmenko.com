import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Clip Generator | AI Video Platform',
  description: 'Transform your long-form videos into engaging short-form vertical clips with AI.'
};

export default function ClipGeneratorLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
} 