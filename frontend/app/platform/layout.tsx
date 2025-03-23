import { Sidebar } from '@/components/navigation/Sidebar';
import { TopNav } from '@/components/navigation/TopNav';

export default function PlatformLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-full">
      <Sidebar />
      <div className="flex flex-1 flex-col">
        <TopNav />
        <main className="flex-1 overflow-y-auto bg-neutral-800 p-6">
          {children}
        </main>
      </div>
    </div>
  );
} 