"use client";

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Video, 
  UserSquare, 
  Coins, 
  HelpCircle,
  Upload,
  Plus,
  DollarSign,
  Book
} from 'lucide-react';

export default function PlatformDashboard() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading data
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white">
        <div className="text-xl">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="container mx-auto flex justify-between items-center">
          <Link href="/" className="text-xl font-bold text-blue-400">
            AI Video Editing Platform
          </Link>
          <nav>
            <ul className="flex space-x-6">
              <li><Link href="/platform" className="text-white hover:text-blue-400 transition">Dashboard</Link></li>
              <li><Link href="/platform/videos" className="text-gray-300 hover:text-blue-400 transition">Videos</Link></li>
              <li><Link href="/platform/avatars" className="text-gray-300 hover:text-blue-400 transition">Avatars</Link></li>
              <li><Link href="/platform/translation" className="text-gray-300 hover:text-blue-400 transition">Translation</Link></li>
              <li><Link href="/platform/clips" className="text-gray-300 hover:text-blue-400 transition">Clip Generator</Link></li>
            </ul>
          </nav>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-300">60 tokens remaining</span>
            <Button variant="outline" className="border-blue-500 text-blue-400">Account</Button>
          </div>
        </div>
      </header>
      
      <main className="container mx-auto p-6">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">Welcome to your dashboard</h2>
          <p className="text-gray-300">Manage your videos, avatars, and more from here.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <Video className="mr-2 h-5 w-5" />
                Videos
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 mb-4">Upload and edit your videos with AI technology.</p>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                <Upload className="mr-2 h-4 w-4" /> Upload Video
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <UserSquare className="mr-2 h-5 w-5" />
                Avatars
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 mb-4">Create AI-powered avatars from your videos.</p>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                <Plus className="mr-2 h-4 w-4" /> Create Avatar
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <Coins className="mr-2 h-5 w-5" />
                Tokens
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 mb-4">You have 60 tokens available for processing.</p>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                <DollarSign className="mr-2 h-4 w-4" /> Buy More Tokens
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <HelpCircle className="mr-2 h-5 w-5" />
                Help
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 mb-4">Need assistance with the platform features?</p>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                <Book className="mr-2 h-4 w-4" /> View Tutorials
              </Button>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}