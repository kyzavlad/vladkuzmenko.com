"use client";

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import Image from 'next/image';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from "@/components/ui/dialog";
import { X } from 'lucide-react';

const images = [
  {
    src: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=500&q=80",
    alt: "AI Analytics Dashboard",
    title: "AI Analytics Dashboard",
    description: "Real-time insights and predictive analytics for business performance monitoring."
  },
  {
    src: "https://images.unsplash.com/photo-1573164713988-8665fc963095?w=800&h=500&q=80",
    alt: "Customer Support Automation",
    title: "Customer Support Automation",
    description: "AI-powered chatbots handling customer inquiries with natural language understanding."
  },
  {
    src: "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=800&h=500&q=80",
    alt: "AI Content Generation",
    title: "AI Content Generation",
    description: "Automated content creation for marketing materials, blogs, and social media."
  },
  {
    src: "https://images.unsplash.com/photo-1563986768609-322da13575f3?w=800&h=500&q=80",
    alt: "Email Marketing Automation",
    title: "Email Marketing Automation",
    description: "Personalized email campaigns with AI-driven targeting and optimization."
  },
  {
    src: "https://images.unsplash.com/photo-1499750310107-5fef28a66643?w=800&h=500&q=80",
    alt: "Workflow Automation",
    title: "Workflow Automation",
    description: "Streamlined business processes with intelligent task routing and approval flows."
  },
  {
    src: "https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1?w=800&h=500&q=80",
    alt: "Voice Assistant Integration",
    title: "Voice Assistant Integration",
    description: "Natural voice interactions for hands-free operation and customer service."
  },
  {
    src: "https://images.unsplash.com/photo-1589578527966-fdac0f44566c?w=800&h=500&q=80",
    alt: "Booking System Automation",
    title: "Booking System Automation",
    description: "Intelligent scheduling and resource management for appointments and reservations."
  }
];

export function ThreeDPhotoCarousel() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const [rotateY, setRotateY] = useState(0);
  const carouselRef = useRef<HTMLDivElement>(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [selectedImage, setSelectedImage] = useState<typeof images[0] | null>(null);

  // Handle mouse/touch events for rotation
  const handleDragStart = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDragging(true);
    setAutoRotate(false);
    
    // Get the starting position
    if ('touches' in e) {
      setStartX(e.touches[0].clientX);
    } else {
      setStartX(e.clientX);
    }
  };

  const handleDragMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDragging) return;
    
    // Calculate the drag distance
    let currentX;
    if ('touches' in e) {
      currentX = e.touches[0].clientX;
    } else {
      currentX = e.clientX;
    }
    
    const deltaX = currentX - startX;
    
    // Update the rotation based on drag distance
    setRotateY(prevRotateY => prevRotateY + deltaX * 0.5);
    setStartX(currentX);
    
    // Calculate the active index based on rotation
    const itemAngle = 360 / images.length;
    const normalizedRotation = ((rotateY % 360) + 360) % 360;
    const newIndex = Math.round(normalizedRotation / itemAngle) % images.length;
    
    if (newIndex !== activeIndex) {
      setActiveIndex(newIndex);
    }
  };

  const handleDragEnd = () => {
    setIsDragging(false);
  };

  // Auto-rotation effect
  useEffect(() => {
    if (!autoRotate) return;
    
    const interval = setInterval(() => {
      setRotateY(prev => prev + 0.5);
      
      // Calculate the active index based on rotation
      const itemAngle = 360 / images.length;
      const normalizedRotation = ((rotateY % 360) + 360) % 360;
      const newIndex = Math.round(normalizedRotation / itemAngle) % images.length;
      
      if (newIndex !== activeIndex) {
        setActiveIndex(newIndex);
      }
    }, 50);
    
    return () => clearInterval(interval);
  }, [autoRotate, activeIndex, rotateY]);

  // Handle image click
  const handleImageClick = (image: typeof images[0]) => {
    setSelectedImage(image);
  };

  return (
    <div className="relative h-[500px] w-full overflow-hidden">
      {/* 3D Carousel */}
      <div 
        ref={carouselRef}
        className="w-full h-full flex items-center justify-center"
        onMouseDown={handleDragStart}
        onMouseMove={handleDragMove}
        onMouseUp={handleDragEnd}
        onMouseLeave={handleDragEnd}
        onTouchStart={handleDragStart}
        onTouchMove={handleDragMove}
        onTouchEnd={handleDragEnd}
      >
        <div 
          className="relative w-[200px] h-[200px] md:w-[300px] md:h-[300px] transform-style-3d"
          style={{ 
            transformStyle: 'preserve-3d',
            transform: `rotateY(${rotateY}deg)`,
            transition: isDragging ? 'none' : 'transform 0.5s ease-out'
          }}
        >
          {images.map((image, index) => {
            const angle = (360 / images.length) * index;
            const radius = 250;
            
            return (
              <Dialog key={index}>
                <DialogTrigger asChild>
                  <div
                    className={cn(
                      "absolute w-[200px] h-[150px] md:w-[300px] md:h-[200px] cursor-pointer",
                      "rounded-lg overflow-hidden border-2 transition-all duration-300",
                      activeIndex === index ? "border-brand shadow-lg scale-110 z-10" : "border-border/50"
                    )}
                    style={{
                      transform: `rotateY(${angle}deg) translateZ(${radius}px)`,
                      transformStyle: 'preserve-3d',
                      backfaceVisibility: 'hidden',
                    }}
                    onClick={() => handleImageClick(image)}
                  >
                    <Image
                      src={image.src}
                      alt={image.alt}
                      fill
                      className="object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent flex flex-col justify-end p-4">
                      <h3 className="text-white text-sm md:text-base font-bold">{image.title}</h3>
                    </div>
                  </div>
                </DialogTrigger>
                <DialogContent className="sm:max-w-2xl">
                  <DialogHeader>
                    <DialogTitle>{image.title}</DialogTitle>
                  </DialogHeader>
                  <div className="relative w-full h-[300px] md:h-[400px] rounded-lg overflow-hidden mb-4">
                    <Image
                      src={image.src}
                      alt={image.alt}
                      fill
                      className="object-cover"
                    />
                  </div>
                  <p className="text-muted-foreground">{image.description}</p>
                </DialogContent>
              </Dialog>
            );
          })}
        </div>
      </div>
      
      {/* Instruction text */}
      <div className="absolute bottom-4 left-0 right-0 text-center text-sm text-muted-foreground">
        Drag to rotate • Click image to view details
      </div>
    </div>
  );
}