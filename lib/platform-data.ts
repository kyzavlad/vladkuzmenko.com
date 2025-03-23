// Platform data types
export interface Campus {
  id: string;
  name: string;
  icon: React.ReactNode;
  onlineCount: number;
  status: "active" | "coming-soon";
}

export interface Course {
  id: string;
  title: string;
  description: string;
  lessons: Array<{
    id: string;
    title: string;
    duration: string;
    videoUrl: string;
  }>;
}

export interface ChatMessage {
  id: string;
  user: {
    name: string;
    avatar: string;
  };
  content: string;
  timestamp: string;
}

export interface NewsItem {
  id: string;
  title: string;
  meta: string;
  description: string;
  icon: React.ReactNode;
  status: string;
  tags: string[];
  colSpan?: number;
  hasPersistentHover?: boolean;
  content: string;
  comments: Array<{
    id: string;
    user: {
      name: string;
      avatar: string;
    };
    content: string;
    timestamp: string;
    likes?: number;
  }>;
}

export interface Task {
  id: string;
  title: string;
  time?: string;
  frequency: string;
  completed: boolean;
}

// Export data (moved from CampusDashboard.tsx)
export const campusData = {
  // ... existing campusData object
};

export const campuses = [
  // ... existing campuses array
];

export const newsItems = [
  // ... existing newsItems array
];

export const initialTasks = [
  // ... existing initialTasks array
];