import React from 'react';
import { Analytics } from '@vercel/analytics/react';
import { SpeedInsights } from '@vercel/speed-insights/next';
import { PostHog } from 'posthog-js';
import { hotjar } from 'react-hotjar';
import { getCLS, getFID, getLCP } from 'web-vitals';

// Initialize analytics services
export const initializeAnalytics = () => {
  // Initialize PostHog for product analytics
  if (typeof window !== 'undefined') {
    PostHog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY || '', {
      api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://app.posthog.com',
      loaded: (posthog) => {
        if (process.env.NODE_ENV === 'development') posthog.opt_out_capturing();
      },
    });
  }

  // Initialize Hotjar for heatmaps and user behavior
  if (typeof window !== 'undefined') {
    hotjar.initialize(
      process.env.NEXT_PUBLIC_HOTJAR_ID || '',
      process.env.NEXT_PUBLIC_HOTJAR_VERSION || ''
    );
  }
};

// Performance monitoring
export const trackPerformance = (metric: string, value: number) => {
  if (typeof window !== 'undefined') {
    // Send to analytics service
    PostHog.capture('performance_metric', {
      metric,
      value,
      timestamp: new Date().toISOString(),
    });

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`Performance metric: ${metric} = ${value}`);
    }
  }
};

// Error tracking
export const trackError = (error: Error, context?: Record<string, any>) => {
  if (typeof window !== 'undefined') {
    PostHog.capture('error', {
      error: {
        message: error.message,
        stack: error.stack,
        name: error.name,
      },
      context,
      timestamp: new Date().toISOString(),
    });

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('Error tracked:', error, context);
    }
  }
};

// User journey tracking
export const trackUserJourney = (step: string, data?: Record<string, any>) => {
  if (typeof window !== 'undefined') {
    PostHog.capture('user_journey', {
      step,
      data,
      timestamp: new Date().toISOString(),
    });
  }
};

// Feature usage tracking
export const trackFeatureUsage = (feature: string, data?: Record<string, any>) => {
  if (typeof window !== 'undefined') {
    PostHog.capture('feature_usage', {
      feature,
      data,
      timestamp: new Date().toISOString(),
    });
  }
};

// A/B testing
export const getExperimentVariant = (experimentName: string): string => {
  if (typeof window !== 'undefined') {
    return PostHog.getFeatureFlag(experimentName) || 'control';
  }
  return 'control';
};

interface AnalyticsProviderProps {
  children: React.ReactNode;
}

// Analytics component
export const AnalyticsProvider: React.FC<AnalyticsProviderProps> = ({ children }) => {
  return (
    <>
      <Analytics />
      <SpeedInsights />
      {children}
    </>
  );
};

// Performance monitoring hooks
export const usePerformanceMonitoring = () => {
  if (typeof window !== 'undefined') {
    // Monitor Core Web Vitals
    getCLS((metric) => trackPerformance('CLS', metric.value));
    getFID((metric) => trackPerformance('FID', metric.value));
    getLCP((metric) => trackPerformance('LCP', metric.value));

    // Monitor network performance
    if ('performance' in window) {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.entryType === 'navigation') {
            trackPerformance('TTFB', entry.responseStart - entry.requestStart);
            trackPerformance('DOMContentLoaded', entry.domContentLoadedEventEnd - entry.navigationStart);
            trackPerformance('Load', entry.loadEventEnd - entry.navigationStart);
          }
        });
      });

      observer.observe({ entryTypes: ['navigation'] });
    }
  }
};

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

// Error boundary component
export class AnalyticsErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    trackError(error, {
      componentStack: errorInfo.componentStack,
    });
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return <div>Something went wrong. Please try again later.</div>;
    }

    return this.props.children;
  }
} 