import './globals.css';
import { Inter } from 'next/font/google';
import { Providers } from './providers';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata = {
  title: 'AI Video Platform',
  description: 'Advanced AI-powered video editing and processing platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} min-h-screen`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
} 