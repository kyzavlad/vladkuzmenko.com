'use client';

import React, { useRef, useEffect, useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { 
  FiStar, 
  FiClock, 
  FiPlay,
  FiChevronLeft,
  FiChevronRight,
  FiArrowRight
} from 'react-icons/fi';

interface Template {
  id: string;
  title: string;
  description: string;
  thumbnail: string;
  category: string;
  duration: string;
  rating: number;
  href: string;
}

const TemplateCard = ({ template }: { template: Template }) => {
  return (
    <motion.div
      className="group relative bg-neutral-400 rounded-lg overflow-hidden shadow-elevation-1 hover:shadow-elevation-2 transition-shadow"
      whileHover={{ y: -5 }}
      transition={{ duration: 0.2 }}
    >
      {/* Thumbnail */}
      <div className="aspect-video relative overflow-hidden">
        <img 
          src={template.thumbnail} 
          alt={template.title} 
          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
        />
        
        {/* Play button overlay */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-black bg-opacity-40">
          <div className="p-3 rounded-full bg-primary text-white">
            <FiPlay size={24} />
          </div>
        </div>
        
        {/* Category badge */}
        <div className="absolute top-2 left-2">
          <span className="px-2 py-1 text-xs font-medium bg-neutral-500 bg-opacity-75 backdrop-blur-sm text-white rounded-md">
            {template.category}
          </span>
        </div>
        
        {/* Duration badge */}
        <div className="absolute bottom-2 right-2">
          <span className="px-2 py-1 text-xs font-medium bg-neutral-500 bg-opacity-75 backdrop-blur-sm text-white rounded-md flex items-center">
            <FiClock size={12} className="mr-1" />
            {template.duration}
          </span>
        </div>
      </div>
      
      {/* Content */}
      <div className="p-3">
        <Link href={template.href} className="block w-full">
          <h3 className="font-medium text-neutral-100 truncate group-hover:text-primary transition-colors">
            {template.title}
          </h3>
          <p className="mt-1 text-sm text-neutral-200 line-clamp-2 h-10">
            {template.description}
          </p>
          
          {/* Rating */}
          <div className="flex items-center mt-2">
            <div className="flex items-center">
              {[1, 2, 3, 4, 5].map((star) => (
                <FiStar 
                  key={star}
                  size={14} 
                  className={`${
                    star <= template.rating ? 'text-yellow-500 fill-yellow-500' : 'text-neutral-300'
                  }`}
                />
              ))}
            </div>
            <span className="ml-2 text-xs text-neutral-200">
              {template.rating.toFixed(1)}
            </span>
          </div>
        </Link>
      </div>
    </motion.div>
  );
};

export default function FeaturedTemplates() {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);

  const templates: Template[] = [
    {
      id: '1',
      title: 'Product Launch Announcement',
      description: 'A professional template for announcing new products and features with dynamic product showcases.',
      thumbnail: 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=800&auto=format&fit=crop',
      category: 'Marketing',
      duration: '1:30',
      rating: 4.8,
      href: '/platform/templates/1'
    },
    {
      id: '2',
      title: 'Corporate Introduction',
      description: 'Present your company, team, and vision with this elegant and professional corporate template.',
      thumbnail: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=800&auto=format&fit=crop',
      category: 'Corporate',
      duration: '2:15',
      rating: 4.5,
      href: '/platform/templates/2'
    },
    {
      id: '3',
      title: 'Social Media Story',
      description: 'Vertical format template optimized for social media stories with animated text and transitions.',
      thumbnail: 'https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?q=80&w=800&auto=format&fit=crop',
      category: 'Social Media',
      duration: '0:30',
      rating: 4.7,
      href: '/platform/templates/3'
    },
    {
      id: '4',
      title: 'Animated Infographics',
      description: 'Present data and statistics with dynamic animated charts and infographics.',
      thumbnail: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=800&auto=format&fit=crop',
      category: 'Education',
      duration: '1:45',
      rating: 4.6,
      href: '/platform/templates/4'
    },
    {
      id: '5',
      title: 'Testimonial Showcase',
      description: 'Highlight customer testimonials and reviews with this professional template.',
      thumbnail: 'https://images.unsplash.com/photo-1557804506-669a67965ba0?q=80&w=800&auto=format&fit=crop',
      category: 'Marketing',
      duration: '1:15',
      rating: 4.9,
      href: '/platform/templates/5'
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
        <h2 className="text-xl font-bold">Featured Templates</h2>
        <Link 
          href="/platform/templates" 
          className="text-primary text-sm hover:underline flex items-center"
        >
          View All Templates
          <FiArrowRight size={16} className="ml-1" />
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
        
        {/* Templates carousel */}
        <div 
          ref={scrollContainerRef}
          className="flex space-x-4 overflow-x-auto pb-4 hide-scrollbar snap-x"
        >
          {templates.map((template) => (
            <div 
              key={template.id} 
              className="w-72 flex-shrink-0 snap-start"
            >
              <TemplateCard template={template} />
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