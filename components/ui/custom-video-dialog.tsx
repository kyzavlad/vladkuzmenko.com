"use client";

import { useState, useRef, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Play, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface CustomVideoDialogProps {
  videoSrc: string;
  title?: string;
  className?: string;
}

export function CustomVideoDialog({
  videoSrc,
  title = "YouTube video player",
  className,
}: CustomVideoDialogProps) {
  const [isVideoOpen, setIsVideoOpen] = useState(false);
  const [glowingDots, setGlowingDots] = useState<Array<{id: number, style: React.CSSProperties}>>([]);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  
  // Generate glowing dots only on client-side to avoid hydration mismatch
  useEffect(() => {
    const dots = Array.from({ length: 20 }, (_, i) => ({
      id: i,
      style: {
        width: `${Math.random() * 4 + 2}px`,
        height: `${Math.random() * 4 + 2}px`,
        top: `${Math.random() * 100}%`,
        left: `${Math.random() * 100}%`,
        animationDuration: `${Math.random() * 3 + 2}s`,
        animationDelay: `${Math.random() * 2}s`,
      }
    }));
    setGlowingDots(dots);
  }, []);
  
  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsVideoOpen(false);
    };
    
    if (isVideoOpen) {
      window.addEventListener('keydown', handleEscape);
    }
    
    return () => {
      window.removeEventListener('keydown', handleEscape);
    };
  }, [isVideoOpen]);

  return (
    <div className={cn("relative w-full", className)}>
      {/* Video thumbnail with play button */}
      <div 
        ref={videoContainerRef}
        className="relative cursor-pointer group overflow-hidden rounded-xl"
        onClick={() => setIsVideoOpen(true)}
      >
        {/* Rainbow border animation */}
        <div className="absolute -inset-[3px] rounded-xl bg-gradient-to-r from-[#ff1f71] via-[#ff7e1f] via-[#ffde1f] via-[#1fdf38] via-[#1f9aff] to-[#df1fff] opacity-80 animate-rainbow" style={{ '--speed': '4s' } as React.CSSProperties}></div>
        
        {/* Video placeholder */}
        <div className="relative z-10 aspect-video bg-black/90 rounded-xl overflow-hidden border border-white/10">
          {/* Animated background gradient */}
          <div className="absolute inset-0 bg-gradient-to-br from-black via-gray-900 to-black opacity-90"></div>
          
          {/* Glowing dots in background - client-side only rendering */}
          <div className="absolute inset-0 overflow-hidden">
            {glowingDots.map((dot) => (
              <div 
                key={dot.id}
                className="absolute rounded-full bg-white/20 animate-pulse"
                style={dot.style}
              />
            ))}
          </div>
          
          <div className="absolute inset-0 flex items-center justify-center">
            {/* Play button with hover effect */}
            <div className="relative group-hover:scale-110 transition-transform duration-500">
              {/* Outer glow */}
              <div className="absolute inset-0 bg-brand/20 rounded-full blur-xl scale-150 opacity-70 group-hover:opacity-100 transition-opacity"></div>
              
              {/* Animated ring */}
              <div className="absolute -inset-4 rounded-full border-2 border-brand/30 animate-pulse opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              
              {/* Play button */}
              <div className="relative bg-gradient-to-br from-brand/80 to-brand rounded-full p-5 shadow-lg shadow-brand/20">
                <Play className="w-10 h-10 text-white fill-white" />
              </div>
            </div>
          </div>
          
          {/* Video title overlay */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-6">
            <p className="text-white font-medium text-lg">{title}</p>
            <p className="text-white/70 text-sm mt-1">Click to play video</p>
          </div>
        </div>
      </div>

      {/* Video modal */}
      <AnimatePresence>
        {isVideoOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
            onClick={() => setIsVideoOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className="relative w-full max-w-5xl aspect-video"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Rainbow border animation */}
              <div className="absolute -inset-[3px] rounded-xl bg-gradient-to-r from-[#ff1f71] via-[#ff7e1f] via-[#ffde1f] via-[#1fdf38] via-[#1f9aff] to-[#df1fff] opacity-80 animate-rainbow" style={{ '--speed': '4s' } as React.CSSProperties}></div>
              
              {/* Glow effect */}
              <div className="absolute -inset-4 bg-brand/20 rounded-xl blur-xl opacity-70"></div>
              
              {/* Close button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setIsVideoOpen(false);
                }}
                className="absolute -top-12 right-0 bg-white/10 hover:bg-white/20 text-white rounded-full p-2 backdrop-blur-md transition-colors z-10"
                aria-label="Close video"
              >
                <X className="w-5 h-5" />
              </button>
              
              {/* Video iframe */}
              <div className="relative z-10 w-full h-full rounded-lg overflow-hidden border border-white/20">
                <iframe
                  src={`${videoSrc}${videoSrc.includes('?') ? '&' : '?'}autoplay=1`}
                  title={title}
                  className="w-full h-full"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  allowFullScreen
                  frameBorder="0"
                ></iframe>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}