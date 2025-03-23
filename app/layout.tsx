import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { ThemeProviderWrapper } from '@/components/theme-provider-wrapper';

const inter = Inter({ subsets: ['latin'] });

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
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className} suppressHydrationWarning>
        <ThemeProviderWrapper>
          {children}
        </ThemeProviderWrapper>
      </body>
    </html>
  );
}