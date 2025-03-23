// Add TypeScript types for the dashboard
interface Window {
  PlatformDashboard?: {
    init: (containerId: string) => void;
  };
}