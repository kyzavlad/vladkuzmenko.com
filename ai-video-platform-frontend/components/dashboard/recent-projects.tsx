'use client';

import React, { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { 
  FiVideo, 
  FiUsers, 
  FiGlobe, 
  FiScissors,
  FiChevronLeft,
  FiChevronRight,
  FiClock,
  FiEdit2,
  FiMoreVertical
} from 'react-icons/fi';
import { formatDate } from '../../lib/utils/formatters';

interface Project {
  id: string;
  name: string;
  type: 'video' | 'avatar' | 'translation' | 'clip';
  thumbnail?: string;
  lastEdited: string;
  progress: number;
  href: string;
}

const ProjectCard = ({ project }: { project: Project }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  
  const getIcon = () => {
    switch (project.type) {
      case 'video': return <FiVideo size={18} />;
      case 'avatar': return <FiUsers size={18} />;
      case 'translation': return <FiGlobe size={18} />;
      case 'clip': return <FiScissors size={18} />;
      default: return <FiVideo size={18} />;
    }
  };
  
  const getTypeLabel = () => {
    switch (project.type) {
      case 'video': return 'Video Project';
      case 'avatar': return 'AI Avatar';
      case 'translation': return 'Translation';
      case 'clip': return 'Clip';
      default: return 'Project';
    }
  };
  
  const getBackgroundColor = () => {
    switch (project.type) {
      case 'video': return 'bg-primary bg-opacity-10';
      case 'avatar': return 'bg-purple bg-opacity-10';
      case 'translation': return 'bg-pink bg-opacity-10';
      case 'clip': return 'bg-orange bg-opacity-10';
      default: return 'bg-primary bg-opacity-10';
    }
  };
  
  const getIconColor = () => {
    switch (project.type) {
      case 'video': return 'text-primary';
      case 'avatar': return 'text-purple';
      case 'translation': return 'text-pink';
      case 'clip': return 'text-orange';
      default: return 'text-primary';
    }
  };

  return (
    <motion.div 
      className="group relative bg-neutral-400 rounded-lg overflow-hidden shadow-elevation-1 hover:shadow-elevation-2 transition-shadow"
      whileHover={{ y: -5 }}
      transition={{ duration: 0.2 }}
    >
      {/* Thumbnail */}
      <div className="aspect-video relative overflow-hidden">
        {project.thumbnail ? (
          <img 
            src={project.thumbnail} 
            alt={project.name} 
            className="w-full h-full object-cover"
          />
        ) : (
          <div className={`w-full h-full flex items-center justify-center ${getBackgroundColor()}`}>
            <div className={getIconColor()}>
              {getIcon()}
            </div>
          </div>
        )}
        
        {/* Progress bar */}
        {project.progress < 100 && (
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-neutral-300">
            <div 
              className="h-full bg-primary" 
              style={{ width: `${project.progress}%` }}
            />
          </div>
        )}
      </div>
      
      {/* Content */}
      <div className="p-3">
        <div className="flex justify-between items-start">
          <Link href={project.href} className="block w-full">
            <h3 className="font-medium text-neutral-100 truncate">{project.name}</h3>
            <div className="flex items-center mt-1">
              <span className={`inline-flex items-center text-xs px-2 py-1 rounded-full ${getBackgroundColor()} ${getIconColor()}`}>
                {getIcon()}
                <span className="ml-1">{getTypeLabel()}</span>
              </span>
            </div>
          </Link>
          
          {/* Menu button */}
          <div className="relative">
            <button 
              className="p-1.5 rounded-full text-neutral-200 hover:bg-neutral-300 transition-colors"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              <FiMoreVertical size={16} />
            </button>
            
            {/* Menu */}
            {isMenuOpen && (
              <div className="absolute top-full right-0 mt-1 w-36 bg-neutral-400 rounded-md shadow-elevation-3 z-10 py-1">
                <Link 
                  href={project.href} 
                  className="flex items-center px-3 py-2 text-sm text-neutral-100 hover:bg-neutral-300 transition-colors"
                >
                  <FiEdit2 size={14} className="mr-2" />
                  <span>Edit</span>
                </Link>
                <button
                  className="w-full flex items-center px-3 py-2 text-sm text-neutral-100 hover:bg-neutral-300 transition-colors"
                  onClick={() => setIsMenuOpen(false)}
                >
                  <FiVideo size={14} className="mr-2" />
                  <span>Preview</span>
                </button>
                <button
                  className="w-full flex items-center px-3 py-2 text-sm text-red-500 hover:bg-neutral-300 transition-colors"
                  onClick={() => setIsMenuOpen(false)}
                >
                  <FiVideo size={14} className="mr-2" />
                  <span>Delete</span>
                </button>
              </div>
            )}
          </div>
        </div>
        
        {/* Last edited */}
        <div className="flex items-center mt-2 text-xs text-neutral-200">
          <FiClock size={12} className="mr-1" />
          <span>Last edited {formatDate(project.lastEdited, { month: 'short', day: 'numeric' })}</span>
        </div>
      </div>
    </motion.div>
  );
};

export default function RecentProjects() {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);

  const projects: Project[] = [
    {
      id: '1',
      name: 'Product Demo Video',
      type: 'video',
      thumbnail: 'https://images.unsplash.com/photo-1626544827763-d516dce335e2?q=80&w=800&auto=format&fit=crop',
      lastEdited: '2023-12-15T14:34:12Z',
      progress: 85,
      href: '/platform/video-editor/project/1'
    },
    {
      id: '2',
      name: 'CEO Avatar for Presentation',
      type: 'avatar',
      lastEdited: '2023-12-14T09:12:45Z',
      progress: 100,
      href: '/platform/avatar-creator/project/2'
    },
    {
      id: '3',
      name: 'Marketing Video Translation',
      type: 'translation',
      thumbnail: 'https://images.unsplash.com/photo-1626785774573-4b799315345d?q=80&w=800&auto=format&fit=crop',
      lastEdited: '2023-12-13T16:45:00Z',
      progress: 60,
      href: '/platform/translation/project/3'
    },
    {
      id: '4',
      name: 'Social Media Highlights',
      type: 'clip',
      thumbnail: 'https://images.unsplash.com/photo-1600132806608-231446b2e7af?q=80&w=800&auto=format&fit=crop',
      lastEdited: '2023-12-12T10:23:15Z',
      progress: 100,
      href: '/platform/clip-generator/project/4'
    },
    {
      id: '5',
      name: 'Training Video For New Hires',
      type: 'video',
      lastEdited: '2023-12-10T11:10:00Z',
      progress: 30,
      href: '/platform/video-editor/project/5'
    }
  ];

  const checkScroll = () => {
    if (!scrollContainerRef.current) return;
    
    const { scrollLeft, scrollWidth, clientWidth } = scrollContainerRef.current;
    setCanScrollLeft(scrollLeft > 0);
    setCanScrollRight(scrollLeft + clientWidth < scrollWidth - 10);
  };

  useEffect(() => {
    const scrollContainer = scrollContainerRef.current;
    if (scrollContainer) {
      scrollContainer.addEventListener('scroll', checkScroll);
      checkScroll();
      
      return () => scrollContainer.removeEventListener('scroll', checkScroll);
    }
  }, []);

  const scroll = (direction: 'left' | 'right') => {
    if (!scrollContainerRef.current) return;
    
    const { clientWidth } = scrollContainerRef.current;
    const scrollAmount = direction === 'left' ? -clientWidth / 2 : clientWidth / 2;
    
    scrollContainerRef.current.scrollBy({
      left: scrollAmount,
      behavior: 'smooth'
    });
  };

  return (
    <div className="mb-8">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Recent Projects</h2>
        <Link 
          href="/platform/projects" 
          className="text-primary text-sm hover:underline flex items-center"
        >
          View All Projects
          <FiChevronRight size={16} className="ml-1" />
        </Link>
      </div>
      
      <div className="relative">
        {/* Scroll buttons */}
        {canScrollLeft && (
          <button
            className="absolute left-0 top-1/2 transform -translate-y-1/2 z-10 bg-neutral-500 bg-opacity-70 backdrop-blur-sm text-white p-2 rounded-r-lg shadow-elevation-2"
            onClick={() => scroll('left')}
          >
            <FiChevronLeft size={20} />
          </button>
        )}
        
        {canScrollRight && (
          <button
            className="absolute right-0 top-1/2 transform -translate-y-1/2 z-10 bg-neutral-500 bg-opacity-70 backdrop-blur-sm text-white p-2 rounded-l-lg shadow-elevation-2"
            onClick={() => scroll('right')}
          >
            <FiChevronRight size={20} />
          </button>
        )}
        
        {/* Projects carousel */}
        <div 
          ref={scrollContainerRef}
          className="flex space-x-4 overflow-x-auto pb-4 hide-scrollbar snap-x"
        >
          {projects.map((project) => (
            <div 
              key={project.id} 
              className="w-64 flex-shrink-0 snap-start"
            >
              <ProjectCard project={project} />
            </div>
          ))}
        </div>
      </div>
      
      {/* CSS for hiding scrollbar but allowing scroll */}
      <style jsx global>{`
        .hide-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
        .hide-scrollbar::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </div>
  );
} 