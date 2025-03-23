import { Inter, Montserrat, Roboto_Mono } from 'next/font/google';
import { Providers } from '@/components/providers';
import '@/styles/globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const montserrat = Montserrat({ subsets: ['latin'], variable: '--font-montserrat' });
const robotoMono = Roboto_Mono({ subsets: ['latin'], variable: '--font-roboto-mono' });

export const metadata = {
  title: 'AI Video Platform',
  description: 'Professional AI-powered video editing platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full">
      <body
        className={`${inter.variable} ${montserrat.variable} ${robotoMono.variable} font-sans antialiased h-full bg-neutral-700 text-neutral-50`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
} 