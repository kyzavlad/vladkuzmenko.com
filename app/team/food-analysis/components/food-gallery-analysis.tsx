'use client';

import { useState } from 'react';
import { Search, ImagePlus, Clock, Bookmark, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface FoodGalleryAnalysisProps {
  isAnalyzing: boolean;
  onAnalyze: () => void;
}

// Mock data for gallery images
const RECENT_IMAGES = [
  { id: 'r1', src: 'https://placehold.co/400x300/333/888?text=Lunch', label: 'Lunch - Yesterday', date: '1 day ago' },
  { id: 'r2', src: 'https://placehold.co/400x300/333/888?text=Breakfast', label: 'Breakfast - Today', date: '6 hours ago' },
  { id: 'r3', src: 'https://placehold.co/400x300/333/888?text=Snack', label: 'Afternoon Snack', date: '3 hours ago' },
  { id: 'r4', src: 'https://placehold.co/400x300/333/888?text=Dinner', label: 'Dinner - Yesterday', date: '1 day ago' },
];

const SAVED_IMAGES = [
  { id: 's1', src: 'https://placehold.co/400x300/333/888?text=Salad', label: 'Favorite Salad', tags: ['healthy', 'lunch'] },
  { id: 's2', src: 'https://placehold.co/400x300/333/888?text=Smoothie', label: 'Protein Smoothie', tags: ['breakfast', 'protein'] },
  { id: 's3', src: 'https://placehold.co/400x300/333/888?text=Meal+Prep', label: 'Weekly Meal Prep', tags: ['prep', 'bulk'] },
  { id: 's4', src: 'https://placehold.co/400x300/333/888?text=Restaurant', label: 'Restaurant Order', tags: ['eating out', 'dinner'] },
  { id: 's5', src: 'https://placehold.co/400x300/333/888?text=Dessert', label: 'Occasional Treat', tags: ['dessert', 'treat'] },
];

export function FoodGalleryAnalysis({ isAnalyzing, onAnalyze }: FoodGalleryAnalysisProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [galleryTab, setGalleryTab] = useState<'recent' | 'saved'>('recent');
  
  // Filter images based on search query
  const filteredRecent = RECENT_IMAGES.filter(img => 
    img.label.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  const filteredSaved = SAVED_IMAGES.filter(img => 
    img.label.toLowerCase().includes(searchQuery.toLowerCase()) || 
    img.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  );
  
  // Handle image selection
  const handleSelectImage = (src: string) => {
    setSelectedImage(src === selectedImage ? null : src);
  };
  
  // Handle analyze button click
  const handleAnalyze = () => {
    if (!selectedImage) return;
    onAnalyze();
  };

  return (
    <div className="food-gallery-analysis">
      {/* Search and filter */}
      <div className="mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input 
            placeholder="Search food images..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-gray-700 border-gray-600 text-white placeholder:text-gray-400"
          />
        </div>
      </div>
      
      {/* Gallery tabs */}
      <Tabs 
        defaultValue="recent" 
        onValueChange={(v) => setGalleryTab(v as 'recent' | 'saved')}
        className="mb-4"
      >
        <TabsList className="grid grid-cols-2 bg-gray-700 text-gray-300">
          <TabsTrigger value="recent" className="flex items-center justify-center">
            <Clock className="h-4 w-4 mr-2" /> Recent
          </TabsTrigger>
          <TabsTrigger value="saved" className="flex items-center justify-center">
            <Bookmark className="h-4 w-4 mr-2" /> Saved
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="recent" className="mt-4">
          {filteredRecent.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>{searchQuery ? 'No matching recent images found' : 'No recent food images'}</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {filteredRecent.map(img => (
                <div 
                  key={img.id}
                  className={`relative cursor-pointer rounded-lg overflow-hidden border-2 transition-colors ${selectedImage === img.src ? 'border-blue-500' : 'border-transparent'}`}
                  onClick={() => handleSelectImage(img.src)}
                >
                  <img src={img.src} alt={img.label} className="w-full aspect-video object-cover" />
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                    <p className="text-white text-sm font-medium">{img.label}</p>
                    <p className="text-gray-300 text-xs">{img.date}</p>
                  </div>
                  {selectedImage === img.src && (
                    <div className="absolute inset-0 bg-blue-500/10 flex items-center justify-center">
                      <div className="h-5 w-5 rounded-full bg-blue-500 flex items-center justify-center">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="saved" className="mt-4">
          {filteredSaved.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Bookmark className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>{searchQuery ? 'No matching saved images found' : 'No saved food images'}</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {filteredSaved.map(img => (
                <div 
                  key={img.id}
                  className={`relative cursor-pointer rounded-lg overflow-hidden border-2 transition-colors ${selectedImage === img.src ? 'border-blue-500' : 'border-transparent'}`}
                  onClick={() => handleSelectImage(img.src)}
                >
                  <img src={img.src} alt={img.label} className="w-full aspect-video object-cover" />
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                    <p className="text-white text-sm font-medium">{img.label}</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {img.tags.map((tag, i) => (
                        <span key={i} className="bg-gray-700/80 text-gray-300 text-xs px-1.5 py-0.5 rounded">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  {selectedImage === img.src && (
                    <div className="absolute inset-0 bg-blue-500/10 flex items-center justify-center">
                      <div className="h-5 w-5 rounded-full bg-blue-500 flex items-center justify-center">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
      
      {/* Upload new option */}
      <Button 
        variant="outline" 
        className="w-full mb-4 border-dashed border-gray-600 text-gray-300 hover:text-white"
      >
        <ImagePlus className="h-4 w-4 mr-2" />
        Upload New Image
      </Button>
      
      {/* Analysis button */}
      <Button
        className="w-full bg-blue-600 hover:bg-blue-700 text-white"
        disabled={!selectedImage || isAnalyzing}
        onClick={handleAnalyze}
      >
        {isAnalyzing ? (
          <span className="flex items-center">
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Analyzing...
          </span>
        ) : (
          <>
            <Zap className="mr-2 h-4 w-4" />
            Analyze Selected Image (1 token)
          </>
        )}
      </Button>
      
      {/* Offline indication */}
      <div className="mt-4 text-center text-xs text-gray-400">
        <p>Previously analyzed images are available offline</p>
      </div>
    </div>
  );
} 