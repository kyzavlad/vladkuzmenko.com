import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'My Clips | Clip Generator | AI Video Platform',
  description: 'Manage, edit, and share your generated short-form clips.'
};

export default function ClipsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
} 