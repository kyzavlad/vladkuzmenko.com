"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { v4 as uuidv4 } from 'uuid';
import { useLocalStorage } from "usehooks-ts";
import { format } from "date-fns";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";

import {
  Dumbbell,
  Building2,
  Bot,
  FileText,
  Mail,
  Clock,
  Crown,
  Maximize2,
  Minimize2,
  MessageSquare,
  Globe,
  Video,
  Star,
  Trophy,
  Plus,
  Trash2,
  CheckCircle2,
  Bell,
  TrendingUp,
  Camera,
  Edit2,
  Send,
  ThumbsUp,
  Reply,
  PlayCircle,
  PauseCircle,
  SkipForward,
  Settings,
  MoreVertical,
  X
} from "lucide-react";

// Types
interface User {
  id: string;
  name: string;
  avatar: string;
  rank: string;
  level: number;
  powerLevel: number;
  score: number;
  achievements: number;
}

interface Task {
  id: string;
  title: string;
  time?: string;
  frequency: "Daily" | "Once";
  completed: boolean;
}

interface ChatMessage {
  id: string;
  userId: string;
  userName: string;
  userAvatar: string;
  content: string;
  timestamp: Date;
  channelId: string;
}

interface NewsComment {
  id: string;
  userId: string;
  content: string;
  timestamp: Date;
  likes: number;
  userName: string;
  userAvatar: string;
}

interface NewsPost {
  id: string;
  title: string;
  content: string;
  author: {
    name: string;
    avatar: string;
  };
  timestamp: Date;
  likes: number;
  comments: NewsComment[];
}

interface Campus {
  id: string;
  name: string;
  icon: React.ReactNode;
  color: string;
  onlineCount: number;
  courses: Course[];
  chatChannels: Array<{
    id: string;
    name: string;
    description: string;
  }>;
  news: NewsPost[];
}

interface Course {
  id: string;
  title: string;
  description: string;
  instructor: {
    name: string;
    avatar: string;
  };
  lessons: Lesson[];
  progress: number;
  thumbnail?: string;
}

interface Lesson {
  id: string;
  title: string;
  description: string;
  videoUrl: string;
  duration: string;
  completed: boolean;
  thumbnail?: string;
}

const initialUser: User = {
  id: uuidv4(),
  name: "Vlad Kuzmenko",
  avatar: "https://github.com/shadcn.png",
  rank: "Gold Rook",
  level: 42,
  powerLevel: 2914,
  score: 89,
  achievements: 15
};

const campuses: Campus[] = [
  {
    id: "fitness",
    name: "Fitness Academy",
    icon: <Dumbbell className="h-6 w-6" />,
    color: "blue",
    courses: [
      {
        id: "fitness-fundamentals",
        title: "Fitness Fundamentals",
        description: "Master the basics of physical fitness",
        instructor: {
          name: "Alex Strong",
          avatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face"
        },
        thumbnail: "https://images.unsplash.com/photo-1517963879433-6ad2b056d712?w=800&h=450&fit=crop",
        lessons: [
          {
            id: "lesson-1",
            title: "Introduction to Fitness",
            description: "Learn the fundamentals of fitness training",
            videoUrl: "https://example.com/video1.mp4",
            thumbnail: "https://images.unsplash.com/photo-1517963879433-6ad2b056d712?w=800&h=450&fit=crop",
            duration: "10:00",
            completed: false
          }
        ],
        progress: 0
      }
    ],
    chatChannels: [
      {
        id: "general",
        name: "General",
        description: "General fitness discussion"
      },
      {
        id: "nutrition",
        name: "Nutrition",
        description: "Nutrition and diet tips"
      },
      {
        id: "training",
        name: "Training",
        description: "Training techniques and tips"
      }
    ],
    news: [],
    onlineCount: 1243
  },
  {
    id: "business",
    name: "Business Academy",
    icon: <Building2 className="h-6 w-6" />,
    color: "purple",
    courses: [],
    chatChannels: [
      {
        id: "entrepreneurship",
        name: "Entrepreneurship",
        description: "Business startup discussions"
      },
      {
        id: "marketing",
        name: "Marketing",
        description: "Marketing strategies"
      },
      {
        id: "sales",
        name: "Sales",
        description: "Sales techniques"
      }
    ],
    news: [],
    onlineCount: 892
  },
  {
    id: "ai",
    name: "AI Academy",
    icon: <Bot className="h-6 w-6" />,
    color: "green",
    courses: [],
    chatChannels: [
      {
        id: "machine-learning",
        name: "Machine Learning",
        description: "ML discussions"
      },
      {
        id: "nlp",
        name: "NLP",
        description: "Natural Language Processing"
      },
      {
        id: "computer-vision",
        name: "Computer Vision",
        description: "Computer Vision topics"
      }
    ],
    news: [],
    onlineCount: 1567
  }
];

export function CampusDashboard() {
  // State
  const [hasAccess] = useLocalStorage("platform-access", false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedCampus, setSelectedCampus] = useState<string>("fitness");
  const [view, setView] = useState<"courses" | "chat" | "news">("courses");
  const [selectedCourse, setSelectedCourse] = useState<string | null>(null);
  const [selectedLesson, setSelectedLesson] = useState<string | null>(null);
  const [selectedChannel, setSelectedChannel] = useState<string>("general");
  const [tasks, setTasks] = useLocalStorage<Task[]>("tasks", []);
  const [messages, setMessages] = useLocalStorage<ChatMessage[]>("chat-messages", []);
  const [user, setUser] = useLocalStorage<User>("user", initialUser);
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [newMessage, setNewMessage] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoProgress, setVideoProgress] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Computed values
  const currentCampus = campuses.find(c => c.id === selectedCampus);
  const currentChannel = currentCampus?.chatChannels.find(c => c.id === selectedChannel);

  // Handlers
  const toggleFullscreen = () => setIsFullscreen(!isFullscreen);

  const handleSendMessage = useCallback((content: string) => {
    if (!content.trim() || !user || !currentCampus || !currentChannel) return;

    const newMessage: ChatMessage = {
      id: uuidv4(),
      userId: user.id,
      userName: user.name,
      userAvatar: user.avatar,
      content,
      timestamp: new Date(),
      channelId: `${currentCampus.id}-${currentChannel.id}`
    };

    setMessages(prev => [...prev, newMessage]);
    setNewMessage("");
  }, [currentCampus, currentChannel, user, setMessages]);

  const handleTaskAdd = useCallback(() => {
    const newTask: Task = {
      id: uuidv4(),
      title: "New Task",
      frequency: "Daily",
      completed: false
    };
    setTasks(prev => [...prev, newTask]);
  }, [setTasks]);

  const handleTaskToggle = useCallback((taskId: string) => {
    setTasks(prev => prev.map(task => 
      task.id === taskId ? { ...task, completed: !task.completed } : task
    ));
  }, [setTasks]);

  const handleTaskDelete = useCallback((taskId: string) => {
    setTasks(prev => prev.filter(task => task.id !== taskId));
  }, [setTasks]);

  const handleProfileUpdate = useCallback((updates: Partial<User>) => {
    setUser(prev => ({ ...prev, ...updates }));
    setIsEditingProfile(false);
  }, [setUser]);

  const handleVideoProgress = useCallback(() => {
    if (videoRef.current) {
      const progress = (videoRef.current.currentTime / videoRef.current.duration) * 100;
      setVideoProgress(progress);
    }
  }, []);

  // Render functions
  const renderUserProfile = () => (
    <div className="p-4 border-b border-white/10">
      <div className="flex items-center gap-3">
        <Avatar className="w-10 h-10">
          <AvatarImage src={user.avatar} alt={user.name} />
          <AvatarFallback>{user.name?.[0] || 'U'}</AvatarFallback>
        </Avatar>
        <div>
          <div className="font-medium">{user.name}</div>
          <div className="text-xs text-muted-foreground">
            Level {user.level} • {user.rank}
          </div>
        </div>
        <Button variant="ghost" size="sm" onClick={() => setIsEditingProfile(true)}>
          <Edit2 className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );

  const renderTaskManager = () => (
    <div className="space-y-2">
      {tasks.map((task) => (
        <div key={task.id} className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className={cn(
              "w-6 h-6 p-0 rounded-full",
              task.completed && "bg-brand/20 text-brand"
            )}
            onClick={() => handleTaskToggle(task.id)}
          >
            <CheckCircle2 className="w-4 h-4" />
          </Button>
          <Input
            value={task.title}
            onChange={(e) => {
              setTasks(prev => prev.map(t =>
                t.id === task.id ? { ...t, title: e.target.value } : t
              ));
            }}
            className="flex-1"
          />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleTaskDelete(task.id)}
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      ))}
      <Button
        variant="ghost"
        size="sm"
        className="w-full justify-start"
        onClick={handleTaskAdd}
      >
        <Plus className="w-4 h-4 mr-2" />
        Add Task
      </Button>
    </div>
  );

  const renderCourses = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {currentCampus?.courses.map((course) => (
        <Card key={course.id} className="p-6">
          {course.thumbnail && (
            <div className="relative aspect-video mb-4 rounded-lg overflow-hidden">
              <img 
                src={course.thumbnail} 
                alt={course.title}
                className="object-cover w-full h-full"
              />
            </div>
          )}
          <div className="flex items-center gap-3 mb-4">
            <Avatar className="w-8 h-8">
              <AvatarImage src={course.instructor.avatar} alt={course.instructor.name} />
              <AvatarFallback>{course.instructor.name[0]}</AvatarFallback>
            </Avatar>
            <div>
              <h3 className="text-xl font-semibold">{course.title}</h3>
              <p className="text-sm text-muted-foreground">{course.instructor.name}</p>
            </div>
          </div>
          <p className="text-muted-foreground mb-4">{course.description}</p>
          <Progress value={course.progress} className="mb-4" />
          <div className="space-y-2">
            {course.lessons.map((lesson) => (
              <Button
                key={lesson.id}
                variant="ghost"
                className="w-full justify-start"
                onClick={() => {
                  setSelectedCourse(course.id);
                  setSelectedLesson(lesson.id);
                }}
              >
                {lesson.completed ? (
                  <CheckCircle2 className="w-4 h-4 mr-2 text-brand" />
                ) : (
                  <PlayCircle className="w-4 h-4 mr-2" />
                )}
                {lesson.title}
                <span className="ml-auto text-xs text-muted-foreground">
                  {lesson.duration}
                </span>
              </Button>
            ))}
          </div>
        </Card>
      ))}
    </div>
  );

  const renderChat = () => (
    <div className="flex h-full">
      <div className="w-64 border-r border-white/10 p-4">
        <h3 className="font-semibold mb-4">Channels</h3>
        <div className="space-y-1">
          {currentCampus?.chatChannels.map((channel) => (
            <Button
              key={channel.id}
              variant="ghost"
              className={cn(
                "w-full justify-start",
                selectedChannel === channel.id
                  ? "bg-brand/20 text-brand"
                  : "hover:bg-white/5"
              )}
              onClick={() => setSelectedChannel(channel.id)}
            >
              # {channel.name}
            </Button>
          ))}
        </div>
      </div>
      
      <div className="flex-1 flex flex-col">
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {messages
              .filter(m => m.channelId === `${currentCampus?.id}-${selectedChannel}`)
              .map((message) => (
                <div key={message.id} className="flex items-start gap-3">
                  <Avatar className="w-8 h-8">
                    <AvatarImage src={message.userAvatar} alt={message.userName} />
                    <AvatarFallback>{message.userName[0]}</AvatarFallback>
                  </Avatar>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{message.userName}</span>
                      <span className="text-xs text-muted-foreground">
                        {format(message.timestamp, "MMM d, yyyy")}
                      </span>
                    </div>
                    <p className="mt-1">{message.content}</p>
                  </div>
                </div>
              ))}
          </div>
        </ScrollArea>
        
        <div className="p-4 border-t border-white/10">
          <div className="flex gap-2">
            <Input
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder={`Message #${currentChannel?.name || 'general'}`}
              onKeyPress={(e) => {
                if (e.key === "Enter") {
                  handleSendMessage(newMessage);
                }
              }}
            />
            <Button onClick={() => handleSendMessage(newMessage)}>
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderNews = () => (
    <div className="space-y-6">
      {currentCampus?.news.map((post) => (
        <Card key={post.id} className="p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <Avatar>
                <AvatarImage
                  src={post.author.avatar}
                  alt={post.author.name}
                />
                <AvatarFallback>
                  {post.author.name[0]}
                </AvatarFallback>
              </Avatar>
              <div>
                <h3 className="font-semibold">{post.title}</h3>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span>{post.author.name}</span>
                  <span>•</span>
                  <span>
                    {format(post.timestamp, "MMM d, yyyy")}
                  </span>
                </div>
              </div>
            </div>
            <Button variant="ghost" size="sm">
              <MoreVertical className="w-4 h-4" />
            </Button>
          </div>
          
          <p className="mt-4">{post.content}</p>
          
          <div className="flex items-center gap-4 mt-4">
            <Button variant="ghost" size="sm">
              <ThumbsUp className="w-4 h-4 mr-2" />
              {post.likes}
            </Button>
            <Button variant="ghost" size="sm">
              <Reply className="w-4 h-4 mr-2" />
              Reply
            </Button>
          </div>
          
          <div className="mt-4 space-y-4">
            {post.comments.map((comment) => (
              <div key={comment.id} className="flex items-start gap-3">
                <Avatar className="w-6 h-6">
                  <AvatarImage
                    src={comment.userAvatar}
                    alt={comment.userName}
                  />
                  <AvatarFallback>
                    {comment.userName[0]}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium">
                      {comment.userName}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {format(comment.timestamp, "MMM d, yyyy")}
                    </span>
                  </div>
                  <p className="text-sm mt-1">{comment.content}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  );

  return (
    <div className={cn(
      "bg-black transition-all duration-300",
      isFullscreen ? "fixed inset-0 z-50" : "min-h-screen"
    )}>
      <div className="flex h-full">
        {/* Left sidebar */}
        <div className="w-[72px] bg-[#1e2124] flex flex-col items-center py-3 gap-2">
          {campuses.map((campus) => (
            <button
              key={campus.id}
              className={cn(
                "w-12 h-12 rounded-lg flex items-center justify-center transition-colors",
                selectedCampus === campus.id
                  ? `bg-${campus.color}-500/20 text-${campus.color}-500`
                  : "hover:bg-white/5 text-white/60"
              )}
              onClick={() => setSelectedCampus(campus.id)}
            >
              {campus.icon}
            </button>
          ))}
        </div>

        {/* Main content area */}
        <div className="flex-1 bg-[#313338] p-8">
          {/* Top navigation */}
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-6">
              <h2 className="text-xl font-semibold">{currentCampus?.name}</h2>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  className={cn(
                    "transition-colors",
                    view === "courses" && "text-brand"
                  )}
                  onClick={() => setView("courses")}
                >
                  <Video className="w-4 h-4 mr-2" />
                  Courses
                </Button>
                <Button
                  variant="ghost"
                  className={cn(
                    "transition-colors",
                    view === "chat" && "text-brand"
                  )}
                  onClick={() => setView("chat")}
                >
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Chat
                </Button>
                <Button
                  variant="ghost"
                  className={cn(
                    "transition-colors",
                    view === "news" && "text-brand"
                  )}
                  onClick={() => setView("news")}
                >
                  <Globe className="w-4 h-4 mr-2" />
                  News
                </Button>
              </div>
            </div>
            <Button variant="ghost" onClick={toggleFullscreen}>
              {isFullscreen ? (
                <Minimize2 className="w-4 h-4" />
              ) : (
                <Maximize2 className="w-4 h-4" />
              )}
            </Button>
          </div>

          {/* Dynamic content based on view */}
          {view === "courses" && renderCourses()}
          {view === "chat" && renderChat()}
          {view === "news" && renderNews()}
        </div>

        {/* Right sidebar */}
        <div className="w-[320px] bg-[#2f3136] flex flex-col">
          {/* User profile */}
          {renderUserProfile()}
          
          {/* Task list */}
          <div className="flex-1 p-4">
            <h3 className="font-semibold mb-4">Daily Tasks</h3>
            {renderTaskManager()}
          </div>
        </div>
      </div>
    </div>
  );
}