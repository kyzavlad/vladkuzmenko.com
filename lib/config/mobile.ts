import { useEffect, useState } from 'react';

export interface MobileConfig {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  screenWidth: number;
  screenHeight: number;
  orientation: 'portrait' | 'landscape';
  hasTouch: boolean;
  reducedMotion: boolean;
  networkType: string;
  isOffline: boolean;
}

const MOBILE_BREAKPOINT = 768;
const TABLET_BREAKPOINT = 1024;

export const useMobileConfig = (): MobileConfig => {
  const [config, setConfig] = useState<MobileConfig>({
    isMobile: false,
    isTablet: false,
    isDesktop: false,
    screenWidth: 0,
    screenHeight: 0,
    orientation: 'portrait',
    hasTouch: false,
    reducedMotion: false,
    networkType: 'unknown',
    isOffline: false,
  });

  useEffect(() => {
    const updateConfig = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const isMobile = width < MOBILE_BREAKPOINT;
      const isTablet = width >= MOBILE_BREAKPOINT && width < TABLET_BREAKPOINT;
      const isDesktop = width >= TABLET_BREAKPOINT;

      // Check for touch support
      const hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

      // Check for reduced motion preference
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

      // Get network information
      const networkType = (navigator as any).connection?.type || 'unknown';
      const isOffline = !navigator.onLine;

      setConfig({
        isMobile,
        isTablet,
        isDesktop,
        screenWidth: width,
        screenHeight: height,
        orientation: width > height ? 'landscape' : 'portrait',
        hasTouch,
        reducedMotion,
        networkType,
        isOffline,
      });
    };

    // Initial update
    updateConfig();

    // Add event listeners
    window.addEventListener('resize', updateConfig);
    window.addEventListener('online', updateConfig);
    window.addEventListener('offline', updateConfig);

    // Cleanup
    return () => {
      window.removeEventListener('resize', updateConfig);
      window.removeEventListener('online', updateConfig);
      window.removeEventListener('offline', updateConfig);
    };
  }, []);

  return config;
};

// Mobile-specific utility functions
export const mobileUtils = {
  // Check if device supports camera
  hasCamera: async (): Promise<boolean> => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.some(device => device.kind === 'videoinput');
    } catch {
      return false;
    }
  },

  // Request camera permission
  requestCameraPermission: async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach(track => track.stop());
      return true;
    } catch {
      return false;
    }
  },

  // Check if device supports offline storage
  supportsOfflineStorage: (): boolean => {
    return 'serviceWorker' in navigator && 'caches' in window;
  },

  // Register service worker for offline functionality
  registerServiceWorker: async (): Promise<boolean> => {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js');
        return registration.active !== null;
      } catch {
        return false;
      }
    }
    return false;
  },

  // Optimize image loading for mobile
  optimizeImage: (url: string, width: number): string => {
    // Add image optimization parameters based on the CDN being used
    // This is an example for a generic CDN
    return `${url}?w=${width}&q=80&format=webp`;
  },

  // Handle mobile-specific gestures
  handleSwipe: (
    element: HTMLElement,
    callbacks: {
      onSwipeLeft?: () => void;
      onSwipeRight?: () => void;
      onSwipeUp?: () => void;
      onSwipeDown?: () => void;
    }
  ) => {
    let touchStartX = 0;
    let touchStartY = 0;
    let touchEndX = 0;
    let touchEndY = 0;

    const handleTouchStart = (e: TouchEvent) => {
      touchStartX = e.touches[0].clientX;
      touchStartY = e.touches[0].clientY;
    };

    const handleTouchEnd = (e: TouchEvent) => {
      touchEndX = e.changedTouches[0].clientX;
      touchEndY = e.changedTouches[0].clientY;

      const deltaX = touchEndX - touchStartX;
      const deltaY = touchEndY - touchStartY;

      // Minimum swipe distance
      const minSwipeDistance = 50;

      if (Math.abs(deltaX) > Math.abs(deltaY)) {
        if (Math.abs(deltaX) > minSwipeDistance) {
          if (deltaX > 0 && callbacks.onSwipeRight) {
            callbacks.onSwipeRight();
          } else if (deltaX < 0 && callbacks.onSwipeLeft) {
            callbacks.onSwipeLeft();
          }
        }
      } else {
        if (Math.abs(deltaY) > minSwipeDistance) {
          if (deltaY > 0 && callbacks.onSwipeDown) {
            callbacks.onSwipeDown();
          } else if (deltaY < 0 && callbacks.onSwipeUp) {
            callbacks.onSwipeUp();
          }
        }
      }
    };

    element.addEventListener('touchstart', handleTouchStart);
    element.addEventListener('touchend', handleTouchEnd);

    return () => {
      element.removeEventListener('touchstart', handleTouchStart);
      element.removeEventListener('touchend', handleTouchEnd);
    };
  },

  // Handle mobile-specific scroll behavior
  handleScroll: (
    element: HTMLElement,
    callbacks: {
      onScroll?: () => void;
      onScrollEnd?: () => void;
    }
  ) => {
    let scrollTimeout: NodeJS.Timeout;

    const handleScroll = () => {
      if (callbacks.onScroll) {
        callbacks.onScroll();
      }

      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        if (callbacks.onScrollEnd) {
          callbacks.onScrollEnd();
        }
      }, 150);
    };

    element.addEventListener('scroll', handleScroll);

    return () => {
      element.removeEventListener('scroll', handleScroll);
      clearTimeout(scrollTimeout);
    };
  },
}; 