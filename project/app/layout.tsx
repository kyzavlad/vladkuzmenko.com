import './globals.css';
import type { Metadata } from 'next';
import { ThemeProviderWrapper } from '@/components/theme-provider-wrapper';

export const metadata: Metadata = {
  title: 'VladKuzmenko.com - AI Automation Solutions',
  description: 'Transform your business with AI-powered automation solutions for customer support, email marketing, and content creation.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning className="font-sans">
      <body suppressHydrationWarning>
        <ThemeProviderWrapper>
          {children}
        </ThemeProviderWrapper>
      </body>
    </html>
  );
}