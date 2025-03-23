import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Video Editor | AI Video Platform',
  description: 'Edit videos with AI-powered features and enhancements'
};

export default function VideoEditorLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
} 