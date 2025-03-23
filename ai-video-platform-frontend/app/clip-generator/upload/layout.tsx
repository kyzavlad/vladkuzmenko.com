import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Upload Videos | Clip Generator | AI Video Platform',
  description: 'Upload your long-form videos to generate short-form vertical clips with AI.'
};

export default function UploadLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
} 