import { DashboardStats } from '@/components/dashboard/DashboardStats';
import { RecentProjects } from '@/components/dashboard/RecentProjects';
import { TokenUsage } from '@/components/dashboard/TokenUsage';

export default function PlatformPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="font-heading text-3xl font-bold">Dashboard</h1>
      </div>

      <DashboardStats />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <RecentProjects />
        <TokenUsage />
      </div>
    </div>
  );
} 