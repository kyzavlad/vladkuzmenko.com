import React from 'react';
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

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto p-6">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">Welcome to your dashboard</h2>
          <p className="text-gray-300">Manage your videos, avatars, and more from here.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-blue-500 transition-all duration-300">
            <div className="flex items-center text-blue-400 mb-4">
              <Video className="mr-2 h-5 w-5" />
              <h3 className="text-xl font-semibold">Videos</h3>
            </div>
            <p className="text-gray-300 mb-4">Upload and edit your videos with AI technology.</p>
            <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded flex items-center justify-center">
              <Upload className="mr-2 h-4 w-4" /> Upload Video
            </button>
          </div>
          
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-blue-500 transition-all duration-300">
            <div className="flex items-center text-blue-400 mb-4">
              <UserSquare className="mr-2 h-5 w-5" />
              <h3 className="text-xl font-semibold">Avatars</h3>
            </div>
            <p className="text-gray-300 mb-4">Create AI-powered avatars from your videos.</p>
            <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded flex items-center justify-center">
              <Plus className="mr-2 h-4 w-4" /> Create Avatar
            </button>
          </div>
          
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-blue-500 transition-all duration-300">
            <div className="flex items-center text-blue-400 mb-4">
              <Coins className="mr-2 h-5 w-5" />
              <h3 className="text-xl font-semibold">Tokens</h3>
            </div>
            <p className="text-gray-300 mb-4">You have 60 tokens available for processing.</p>
            <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded flex items-center justify-center">
              <DollarSign className="mr-2 h-4 w-4" /> Buy More Tokens
            </button>
          </div>
          
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-blue-500 transition-all duration-300">
            <div className="flex items-center text-blue-400 mb-4">
              <HelpCircle className="mr-2 h-5 w-5" />
              <h3 className="text-xl font-semibold">Help</h3>
            </div>
            <p className="text-gray-300 mb-4">Need assistance with the platform features?</p>
            <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded flex items-center justify-center">
              <Book className="mr-2 h-4 w-4" /> View Tutorials
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 