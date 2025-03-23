'use client';

import { useState } from 'react';
import {
  Heart,
  MessageCircle,
  Share2,
  Award,
  Camera,
  BarChart3,
  Activity,
  ThumbsUp,
  Send,
  MoreHorizontal,
  Filter,
  TrendingUp,
  Users,
  Bookmark,
  Clock,
  Lock,
  Globe,
  UserPlus
} from 'lucide-react';
import { format } from 'date-fns';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';

type PostPrivacy = 'public' | 'friends' | 'private';
type ContentType = 'status' | 'workout' | 'achievement' | 'progress-photo' | 'milestone';

interface PostComment {
  id: string;
  userId: string;
  username: string;
  userAvatar?: string;
  text: string;
  timestamp: Date;
  likes: number;
}

interface FeedPost {
  id: string;
  userId: string;
  username: string;
  userAvatar?: string;
  content: string;
  mediaUrls?: string[];
  contentType: ContentType;
  privacy: PostPrivacy;
  timestamp: Date;
  likes: number;
  comments: PostComment[];
  metadata?: {
    workoutName?: string;
    workoutDuration?: number;
    achievementName?: string;
    milestoneTitle?: string;
    progress?: {
      current: number;
      previous: number;
      unit: string;
      metricName: string;
    };
  };
}

interface CommunityFeedProps {
  userId?: string;
  initialFilter?: string;
}

export default function CommunityFeed({ userId = 'u1', initialFilter = 'all' }: CommunityFeedProps) {
  const [activeTab, setActiveTab] = useState<string>('feed');
  const [filter, setFilter] = useState<string>(initialFilter);
  const [newPostContent, setNewPostContent] = useState<string>('');
  const [newPostPrivacy, setNewPostPrivacy] = useState<PostPrivacy>('public');
  const [newPostType, setNewPostType] = useState<ContentType>('status');
  const [expandedComments, setExpandedComments] = useState<string[]>([]);
  const [commentInput, setCommentInput] = useState<Record<string, string>>({});
  
  // Mock feed data
  const [feedPosts, setFeedPosts] = useState<FeedPost[]>([
    {
      id: 'p1',
      userId: 'u2',
      username: 'Jennifer Miller',
      userAvatar: '/avatars/jennifer.png',
      content: 'Just crushed a new PR on my deadlift! 225 lbs for 5 reps. Hard work pays off! 💪',
      contentType: 'workout',
      privacy: 'public',
      timestamp: new Date(2023, 6, 15, 9, 30),
      likes: 24,
      comments: [
        {
          id: 'c1',
          userId: 'u3',
          username: 'Michael Johnson',
          userAvatar: '/avatars/michael.png',
          text: 'That\'s amazing progress, Jennifer! Keep crushing it!',
          timestamp: new Date(2023, 6, 15, 10, 15),
          likes: 3
        }
      ],
      metadata: {
        workoutName: 'Lower Body Power',
        workoutDuration: 65
      }
    },
    {
      id: 'p2',
      userId: 'u4',
      username: 'Robert Chen',
      userAvatar: '/avatars/robert.png',
      content: 'Earned my "30 Day Streak" achievement! Consistency is key to results.',
      contentType: 'achievement',
      privacy: 'public',
      timestamp: new Date(2023, 6, 14, 18, 45),
      likes: 31,
      comments: [],
      metadata: {
        achievementName: '30 Day Streak'
      }
    },
    {
      id: 'p3',
      userId: 'u5',
      username: 'Sarah Wilson',
      userAvatar: '/avatars/sarah.png',
      content: 'Monthly progress update! Down 5 lbs and lost 2 inches off my waist since last month. So happy with my results!',
      mediaUrls: ['/progress/sarah-progress-1.jpg'],
      contentType: 'progress-photo',
      privacy: 'friends',
      timestamp: new Date(2023, 6, 13, 12, 10),
      likes: 45,
      comments: [
        {
          id: 'c2',
          userId: 'u2',
          username: 'Jennifer Miller',
          userAvatar: '/avatars/jennifer.png',
          text: 'You look amazing! What\'s your nutrition plan been like?',
          timestamp: new Date(2023, 6, 13, 12, 30),
          likes: 2
        },
        {
          id: 'c3',
          userId: 'u5',
          username: 'Sarah Wilson',
          userAvatar: '/avatars/sarah.png',
          text: 'Thanks! I\'ve been focusing on protein-rich meals and limiting processed foods. Makes such a difference!',
          timestamp: new Date(2023, 6, 13, 13, 5),
          likes: 1
        }
      ],
      metadata: {
        progress: {
          current: 140,
          previous: 145,
          unit: 'lbs',
          metricName: 'Weight'
        }
      }
    },
    {
      id: 'p4',
      userId: 'u6',
      username: 'David Smith',
      userAvatar: '/avatars/david.png',
      content: 'Just completed the 10K Challenge! It was tough but worth it. Anyone else training for a race?',
      contentType: 'milestone',
      privacy: 'public',
      timestamp: new Date(2023, 6, 10, 8, 20),
      likes: 19,
      comments: [],
      metadata: {
        milestoneTitle: '10K Challenge'
      }
    }
  ]);
  
  // Toggle comment visibility
  const toggleComments = (postId: string) => {
    setExpandedComments(prev => {
      if (prev.includes(postId)) {
        return prev.filter(id => id !== postId);
      } else {
        return [...prev, postId];
      }
    });
  };
  
  // Like a post
  const likePost = (postId: string) => {
    setFeedPosts(prev => prev.map(post => {
      if (post.id === postId) {
        return { ...post, likes: post.likes + 1 };
      }
      return post;
    }));
  };
  
  // Add a comment
  const addComment = (postId: string) => {
    if (!commentInput[postId]?.trim()) return;
    
    const newComment: PostComment = {
      id: `c${Date.now()}`,
      userId,
      username: 'Current User', // In a real app, this would be the logged-in user
      userAvatar: '/avatars/user.png', // In a real app, this would be the user's avatar
      text: commentInput[postId],
      timestamp: new Date(),
      likes: 0
    };
    
    setFeedPosts(prev => prev.map(post => {
      if (post.id === postId) {
        return { 
          ...post, 
          comments: [...post.comments, newComment]
        };
      }
      return post;
    }));
    
    // Clear input
    setCommentInput(prev => ({
      ...prev,
      [postId]: ''
    }));
  };
  
  // Create a new post
  const createPost = () => {
    if (!newPostContent.trim()) return;
    
    const newPost: FeedPost = {
      id: `p${Date.now()}`,
      userId,
      username: 'Current User', // In a real app, this would be the logged-in user
      userAvatar: '/avatars/user.png', // In a real app, this would be the user's avatar
      content: newPostContent,
      contentType: newPostType,
      privacy: newPostPrivacy,
      timestamp: new Date(),
      likes: 0,
      comments: []
    };
    
    setFeedPosts(prev => [newPost, ...prev]);
    setNewPostContent('');
  };
  
  // Get icon for post type
  const getPostTypeIcon = (type: ContentType) => {
    switch (type) {
      case 'workout':
        return <Activity className="h-5 w-5 text-green-500" />;
      case 'achievement':
        return <Award className="h-5 w-5 text-yellow-500" />;
      case 'progress-photo':
        return <Camera className="h-5 w-5 text-pink-500" />;
      case 'milestone':
        return <TrendingUp className="h-5 w-5 text-purple-500" />;
      default:
        return <MessageCircle className="h-5 w-5 text-blue-500" />;
    }
  };
  
  // Get icon for privacy setting
  const getPrivacyIcon = (privacy: PostPrivacy) => {
    switch (privacy) {
      case 'public':
        return <Globe className="h-4 w-4" />;
      case 'friends':
        return <Users className="h-4 w-4" />;
      case 'private':
        return <Lock className="h-4 w-4" />;
    }
  };
  
  // Filter posts based on the selected filter
  const filteredPosts = feedPosts.filter(post => {
    if (filter === 'all') return true;
    if (filter === 'workouts') return post.contentType === 'workout';
    if (filter === 'achievements') return post.contentType === 'achievement';
    if (filter === 'progress') return post.contentType === 'progress-photo';
    if (filter === 'milestones') return post.contentType === 'milestone';
    return true;
  });
  
  return (
    <div className="community-feed">
      <div className="mb-6">
        <h2 className="text-white text-xl font-medium">Community Feed</h2>
        <p className="text-gray-400">Stay connected with your fitness community</p>
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="feed">Feed</TabsTrigger>
          <TabsTrigger value="friends">Friends</TabsTrigger>
          <TabsTrigger value="groups">Groups</TabsTrigger>
        </TabsList>
        
        <TabsContent value="feed" className="space-y-6">
          {/* Create Post Card */}
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-4">
              <div className="flex gap-3 mb-3">
                <Avatar>
                  <AvatarImage src="/avatars/user.png" alt="User" />
                  <AvatarFallback>U</AvatarFallback>
                </Avatar>
                
                <div className="flex-grow">
                  <Textarea 
                    value={newPostContent}
                    onChange={(e) => setNewPostContent(e.target.value)}
                    placeholder="Share your fitness update..."
                    className="bg-gray-750 border-gray-600 min-h-[80px]"
                  />
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <div className="flex gap-2">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm" className="flex gap-1">
                        {getPostTypeIcon(newPostType)}
                        <span className="hidden sm:inline">
                          {newPostType === 'status' ? 'Post Type' : 
                            newPostType === 'workout' ? 'Workout' : 
                            newPostType === 'achievement' ? 'Achievement' : 
                            newPostType === 'progress-photo' ? 'Progress Photo' : 
                            'Milestone'}
                        </span>
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                      <DropdownMenuItem onClick={() => setNewPostType('status')}>
                        <MessageCircle className="h-4 w-4 mr-2 text-blue-500" />
                        <span>Status Update</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setNewPostType('workout')}>
                        <Activity className="h-4 w-4 mr-2 text-green-500" />
                        <span>Workout</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setNewPostType('achievement')}>
                        <Award className="h-4 w-4 mr-2 text-yellow-500" />
                        <span>Achievement</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setNewPostType('progress-photo')}>
                        <Camera className="h-4 w-4 mr-2 text-pink-500" />
                        <span>Progress Photo</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setNewPostType('milestone')}>
                        <TrendingUp className="h-4 w-4 mr-2 text-purple-500" />
                        <span>Milestone</span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm" className="flex gap-1">
                        {getPrivacyIcon(newPostPrivacy)}
                        <span className="hidden sm:inline">
                          {newPostPrivacy === 'public' ? 'Public' : 
                            newPostPrivacy === 'friends' ? 'Friends' : 
                            'Private'}
                        </span>
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                      <DropdownMenuItem onClick={() => setNewPostPrivacy('public')}>
                        <Globe className="h-4 w-4 mr-2" />
                        <span>Public</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setNewPostPrivacy('friends')}>
                        <Users className="h-4 w-4 mr-2" />
                        <span>Friends Only</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setNewPostPrivacy('private')}>
                        <Lock className="h-4 w-4 mr-2" />
                        <span>Private</span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                
                <Button onClick={createPost} disabled={!newPostContent.trim()}>
                  <Send className="mr-2 h-4 w-4" />
                  Post
                </Button>
              </div>
            </CardContent>
          </Card>
          
          {/* Feed Filters */}
          <div className="flex overflow-x-auto gap-2 pb-2">
            <Button 
              variant={filter === 'all' ? 'default' : 'outline'} 
              size="sm"
              onClick={() => setFilter('all')}
            >
              <Filter className="mr-1 h-4 w-4" />
              All Updates
            </Button>
            
            <Button 
              variant={filter === 'workouts' ? 'default' : 'outline'} 
              size="sm"
              onClick={() => setFilter('workouts')}
            >
              <Activity className="mr-1 h-4 w-4" />
              Workouts
            </Button>
            
            <Button 
              variant={filter === 'achievements' ? 'default' : 'outline'} 
              size="sm"
              onClick={() => setFilter('achievements')}
            >
              <Award className="mr-1 h-4 w-4" />
              Achievements
            </Button>
            
            <Button 
              variant={filter === 'progress' ? 'default' : 'outline'} 
              size="sm"
              onClick={() => setFilter('progress')}
            >
              <BarChart3 className="mr-1 h-4 w-4" />
              Progress
            </Button>
            
            <Button 
              variant={filter === 'milestones' ? 'default' : 'outline'} 
              size="sm"
              onClick={() => setFilter('milestones')}
            >
              <TrendingUp className="mr-1 h-4 w-4" />
              Milestones
            </Button>
          </div>
          
          {/* Feed Posts */}
          <div className="space-y-4">
            {filteredPosts.length > 0 ? (
              filteredPosts.map(post => (
                <Card key={post.id} className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    {/* Post Header */}
                    <div className="flex justify-between mb-3">
                      <div className="flex gap-3">
                        <Avatar>
                          <AvatarImage src={post.userAvatar} alt={post.username} />
                          <AvatarFallback>{post.username.charAt(0)}</AvatarFallback>
                        </Avatar>
                        
                        <div>
                          <div className="flex items-center gap-2">
                            <div className="text-white font-medium">{post.username}</div>
                            <Badge 
                              variant="outline" 
                              className="text-xs"
                            >
                              {post.contentType === 'status' ? 'Update' : 
                                post.contentType === 'workout' ? 'Workout' : 
                                post.contentType === 'achievement' ? 'Achievement' : 
                                post.contentType === 'progress-photo' ? 'Progress' : 
                                'Milestone'}
                            </Badge>
                          </div>
                          <div className="text-gray-400 text-xs flex items-center gap-1">
                            {format(post.timestamp, 'MMM d, yyyy • h:mm a')}
                            <span className="px-1">•</span>
                            {getPrivacyIcon(post.privacy)}
                          </div>
                        </div>
                      </div>
                      
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem>
                            <Bookmark className="h-4 w-4 mr-2" />
                            <span>Save Post</span>
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Share2 className="h-4 w-4 mr-2" />
                            <span>Share</span>
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                    
                    {/* Post Content */}
                    <div className="mb-4">
                      <p className="text-white mb-3">{post.content}</p>
                      
                      {/* Post Metadata */}
                      {post.contentType === 'workout' && post.metadata?.workoutName && (
                        <div className="bg-gray-750 p-3 rounded-lg mb-3">
                          <div className="flex items-center gap-2 text-blue-400">
                            <Activity className="h-5 w-5" />
                            <span className="font-medium">{post.metadata.workoutName}</span>
                          </div>
                          {post.metadata.workoutDuration && (
                            <div className="flex items-center gap-1 text-gray-400 text-sm mt-1">
                              <Clock className="h-4 w-4" />
                              <span>{post.metadata.workoutDuration} minutes</span>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {post.contentType === 'achievement' && post.metadata?.achievementName && (
                        <div className="bg-yellow-900/20 border border-yellow-800/40 p-3 rounded-lg mb-3">
                          <div className="flex items-center gap-2 text-yellow-400">
                            <Award className="h-5 w-5" />
                            <span className="font-medium">{post.metadata.achievementName}</span>
                          </div>
                        </div>
                      )}
                      
                      {post.contentType === 'milestone' && post.metadata?.milestoneTitle && (
                        <div className="bg-purple-900/20 border border-purple-800/40 p-3 rounded-lg mb-3">
                          <div className="flex items-center gap-2 text-purple-400">
                            <TrendingUp className="h-5 w-5" />
                            <span className="font-medium">{post.metadata.milestoneTitle}</span>
                          </div>
                        </div>
                      )}
                      
                      {/* Progress Data */}
                      {post.contentType === 'progress-photo' && post.metadata?.progress && (
                        <div className="bg-gray-750 p-3 rounded-lg mb-3">
                          <div className="flex items-center gap-2 text-blue-400 mb-2">
                            <BarChart3 className="h-5 w-5" />
                            <span className="font-medium">{post.metadata.progress.metricName} Progress</span>
                          </div>
                          <div className="flex items-center justify-between text-sm">
                            <div className="text-gray-400">Previous: {post.metadata.progress.previous} {post.metadata.progress.unit}</div>
                            <div className="text-gray-200">→</div>
                            <div className="text-green-400">Current: {post.metadata.progress.current} {post.metadata.progress.unit}</div>
                          </div>
                        </div>
                      )}
                      
                      {/* Media */}
                      {post.mediaUrls && post.mediaUrls.length > 0 && (
                        <div className="mt-3">
                          <div className="grid grid-cols-2 gap-2">
                            {post.mediaUrls.map((url, i) => (
                              <div key={i} className="relative aspect-square rounded-lg overflow-hidden bg-gray-900">
                                <img src={url} alt="Post media" className="w-full h-full object-cover" />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    
                    {/* Post Actions */}
                    <div className="flex justify-between items-center border-t border-gray-700 pt-3">
                      <div className="flex gap-4">
                        <button 
                          className="flex items-center gap-1 text-gray-400 hover:text-blue-500 transition-colors"
                          onClick={() => likePost(post.id)}
                        >
                          <ThumbsUp className="h-5 w-5" />
                          <span>{post.likes}</span>
                        </button>
                        
                        <button 
                          className="flex items-center gap-1 text-gray-400 hover:text-blue-500 transition-colors"
                          onClick={() => toggleComments(post.id)}
                        >
                          <MessageCircle className="h-5 w-5" />
                          <span>{post.comments.length}</span>
                        </button>
                        
                        <button className="flex items-center gap-1 text-gray-400 hover:text-blue-500 transition-colors">
                          <Share2 className="h-5 w-5" />
                        </button>
                      </div>
                      
                      <Button variant="ghost" size="sm">
                        <Bookmark className="h-4 w-4 mr-2" />
                        Save
                      </Button>
                    </div>
                    
                    {/* Comments Section */}
                    {expandedComments.includes(post.id) && (
                      <div className="mt-4 pt-3 border-t border-gray-700">
                        <h4 className="text-white font-medium text-sm mb-3">Comments</h4>
                        
                        <div className="space-y-3 mb-4">
                          {post.comments.length > 0 ? (
                            post.comments.map(comment => (
                              <div key={comment.id} className="flex gap-3">
                                <Avatar className="h-8 w-8">
                                  <AvatarImage src={comment.userAvatar} alt={comment.username} />
                                  <AvatarFallback>{comment.username.charAt(0)}</AvatarFallback>
                                </Avatar>
                                
                                <div className="flex-grow">
                                  <div className="bg-gray-750 p-2 rounded-lg">
                                    <div className="text-sm text-white font-medium">{comment.username}</div>
                                    <div className="text-gray-300">{comment.text}</div>
                                  </div>
                                  <div className="flex gap-3 text-xs text-gray-500 mt-1">
                                    <span>{format(comment.timestamp, 'MMM d, yyyy')}</span>
                                    <button className="hover:text-blue-500">Like ({comment.likes})</button>
                                    <button className="hover:text-blue-500">Reply</button>
                                  </div>
                                </div>
                              </div>
                            ))
                          ) : (
                            <div className="text-center text-gray-500 py-2">No comments yet. Be the first to comment!</div>
                          )}
                        </div>
                        
                        <div className="flex gap-2">
                          <Avatar className="h-8 w-8">
                            <AvatarImage src="/avatars/user.png" alt="Current user" />
                            <AvatarFallback>U</AvatarFallback>
                          </Avatar>
                          
                          <div className="flex-grow flex gap-2">
                            <Input 
                              value={commentInput[post.id] || ''}
                              onChange={(e) => setCommentInput(prev => ({
                                ...prev,
                                [post.id]: e.target.value
                              }))}
                              placeholder="Write a comment..."
                              className="bg-gray-750 border-gray-700"
                            />
                            
                            <Button 
                              size="sm" 
                              onClick={() => addComment(post.id)}
                              disabled={!commentInput[post.id]?.trim()}
                            >
                              Post
                            </Button>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))
            ) : (
              <Card className="bg-gray-800 border-gray-700">
                <CardContent className="p-6 text-center">
                  <Filter className="h-16 w-16 text-gray-700 mx-auto mb-4" />
                  <h3 className="text-white text-lg font-medium mb-2">No posts found</h3>
                  <p className="text-gray-400 mb-4">
                    There are no posts matching your current filter.
                    Try selecting a different filter or create a new post!
                  </p>
                  <Button onClick={() => setFilter('all')}>
                    Show All Posts
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>
        
        <TabsContent value="friends" className="space-y-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-white font-medium">Your Fitness Network</h3>
                <Button>
                  <UserPlus className="mr-2 h-4 w-4" />
                  Find Friends
                </Button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Sample friends - in a real app, this would be dynamic */}
                {[
                  { id: 'u2', name: 'Jennifer Miller', avatar: '/avatars/jennifer.png', mutualFriends: 5, recentAchievement: '30 Day Streak' },
                  { id: 'u3', name: 'Michael Johnson', avatar: '/avatars/michael.png', mutualFriends: 3, recentAchievement: 'Marathon Finisher' },
                  { id: 'u4', name: 'Robert Chen', avatar: '/avatars/robert.png', mutualFriends: 8, recentAchievement: 'Weight Loss Goal' },
                  { id: 'u5', name: 'Sarah Wilson', avatar: '/avatars/sarah.png', mutualFriends: 2, recentAchievement: 'Perfect Week' },
                ].map(friend => (
                  <div key={friend.id} className="flex gap-3 bg-gray-750 p-3 rounded-lg">
                    <Avatar>
                      <AvatarImage src={friend.avatar} alt={friend.name} />
                      <AvatarFallback>{friend.name.charAt(0)}</AvatarFallback>
                    </Avatar>
                    
                    <div className="flex-grow">
                      <div className="text-white font-medium">{friend.name}</div>
                      <div className="text-gray-400 text-sm">{friend.mutualFriends} mutual friends</div>
                      {friend.recentAchievement && (
                        <div className="flex items-center gap-1 mt-1 text-yellow-500 text-xs">
                          <Award className="h-3 w-3" />
                          <span>Recent: {friend.recentAchievement}</span>
                        </div>
                      )}
                    </div>
                    
                    <Button variant="outline" size="sm">
                      View Profile
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          
          <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-4">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-full bg-blue-800/50 flex items-center justify-center shrink-0">
                <Activity className="h-5 w-5 text-blue-300" />
              </div>
              
              <div>
                <h3 className="text-white font-medium mb-1">Find Workout Partners</h3>
                <p className="text-gray-300 text-sm mb-3">
                  Connect with people who have similar fitness goals and training schedules.
                  Working out together increases accountability and results!
                </p>
                
                <Button className="bg-blue-600 hover:bg-blue-700">
                  Find Workout Partners
                </Button>
              </div>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="groups" className="space-y-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-6 text-center">
              <Users className="h-16 w-16 text-gray-700 mx-auto mb-4" />
              <h3 className="text-white text-lg font-medium mb-2">Fitness Groups</h3>
              <p className="text-gray-400 mb-4 max-w-md mx-auto">
                Join groups of like-minded individuals with similar fitness interests and goals.
                Participate in challenges, share tips, and stay motivated together.
              </p>
              <Button>
                <Users className="mr-2 h-4 w-4" />
                Browse Groups
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 