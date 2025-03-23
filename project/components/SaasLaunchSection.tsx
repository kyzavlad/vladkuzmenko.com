"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Sparkles, Rocket, Calendar, ArrowRight, Bell } from "lucide-react";
import { cn } from "@/lib/utils";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger,
  DialogFooter,
  DialogClose
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from "next/link";

export function SaasLaunchSection() {
  const [email, setEmail] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [countdown, setCountdown] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0
  });
  const [particles, setParticles] = useState<Array<{top: string, left: string, width: string, height: string, delay: string, duration: string}>>([]);

  // Generate particles on client-side only to avoid hydration mismatch
  useEffect(() => {
    const newParticles = Array(6).fill(0).map(() => ({
      width: `${Math.random() * 8 + 4}px`,
      height: `${Math.random() * 8 + 4}px`,
      top: `${Math.random() * 100}%`,
      left: `${Math.random() * 100}%`,
      delay: `${Math.random() * 5}s`,
      duration: `${Math.random() * 5 + 3}s`
    }));
    setParticles(newParticles);
  }, []);

  // Set launch date to 30 days from now
  const launchDate = new Date();
  launchDate.setDate(launchDate.getDate() + 30);

  // Countdown timer
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      const difference = launchDate.getTime() - now.getTime();
      
      if (difference <= 0) {
        clearInterval(timer);
        return;
      }
      
      const days = Math.floor(difference / (1000 * 60 * 60 * 24));
      const hours = Math.floor((difference % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((difference % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((difference % (1000 * 60)) / 1000);
      
      setCountdown({ days, hours, minutes, seconds });
    }, 1000);
    
    return () => clearInterval(timer);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false);
      setIsSubmitted(true);
      
      // Reset form after 3 seconds
      setTimeout(() => {
        setEmail("");
        setIsSubmitted(false);
      }, 3000);
    }, 1500);
  };

  const formatNumber = (num: number) => {
    return num < 10 ? `0${num}` : num;
  };

  return (
    <div id="saas-launch-section" className="w-full py-20 md:py-32 bg-background relative overflow-hidden">
      {/* Background gradient elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[30%] -right-[10%] w-[40%] h-[40%] bg-gradient-radial from-brand/10 to-transparent opacity-30 blur-3xl"></div>
        <div className="absolute -bottom-[20%] -left-[10%] w-[40%] h-[40%] bg-gradient-radial from-purple-500/10 to-transparent opacity-30 blur-3xl"></div>
      </div>
      
      {/* Animated particles - client-side only */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {particles.map((particle, i) => (
          <div 
            key={i}
            className="absolute rounded-full bg-brand/20 animate-pulse-glow"
            style={{
              width: particle.width,
              height: particle.height,
              top: particle.top,
              left: particle.left,
              animationDelay: particle.delay,
              animationDuration: particle.duration
            }}
          />
        ))}
      </div>
      
      <div className="container mx-auto relative z-10">
        <div className="max-w-4xl mx-auto">
          <div className="flex flex-col items-center text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-brand/10 border border-brand/20 text-brand mb-6">
              <Rocket className="h-4 w-4" />
              <span className="text-sm font-medium">Coming Soon</span>
            </div>
            
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 text-gradient">
              AI Automation Platform
              <span className="relative ml-2 inline-block">
                <span className="relative z-10">2.0</span>
                <span className="absolute -bottom-1 left-0 w-full h-3 bg-brand/20 rounded-sm -z-10"></span>
              </span>
            </h2>
            
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
              Our next-generation AI automation platform is launching soon. Get ready for a revolutionary approach to business automation with advanced AI capabilities, intuitive interfaces, and seamless integration.
            </p>
            
            {/* Countdown timer */}
            <div className="grid grid-cols-4 gap-4 md:gap-6 max-w-xl w-full mb-12">
              {[
                { label: "Days", value: countdown.days },
                { label: "Hours", value: countdown.hours },
                { label: "Minutes", value: countdown.minutes },
                { label: "Seconds", value: countdown.seconds }
              ].map((item, index) => (
                <div key={index} className="flex flex-col items-center">
                  <div className="relative w-full aspect-square">
                    <div className="absolute inset-0 bg-gradient-to-br from-brand/20 to-purple-500/20 rounded-lg blur-sm"></div>
                    <div className="relative flex items-center justify-center w-full h-full bg-background border border-border/50 rounded-lg shadow-sm">
                      <span className="text-2xl md:text-4xl font-bold text-foreground">
                        {formatNumber(item.value)}
                      </span>
                    </div>
                  </div>
                  <span className="text-xs md:text-sm text-muted-foreground mt-2">{item.label}</span>
                </div>
              ))}
            </div>
            
            {/* Feature highlights */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12 w-full">
              <div className="bg-background/50 p-6 rounded-lg border border-border/50 shadow-sm">
                <div className="flex items-center gap-2 mb-3">
                  <span className="p-1.5 rounded-full bg-blue-500/10 text-blue-500">
                    <Sparkles className="h-4 w-4" />
                  </span>
                  <h3 className="text-lg font-semibold">Advanced AI Models</h3>
                </div>
                <p className="text-muted-foreground">
                  Powered by next-generation AI models with enhanced understanding, reasoning, and generation capabilities.
                </p>
              </div>
              
              <div className="bg-background/50 p-6 rounded-lg border border-border/50 shadow-sm">
                <div className="flex items-center gap-2 mb-3">
                  <span className="p-1.5 rounded-full bg-purple-500/10 text-purple-500">
                    <Calendar className="h-4 w-4" />
                  </span>
                  <h3 className="text-lg font-semibold">Early Access</h3>
                </div>
                <p className="text-muted-foreground">
                  Join our early access program to be among the first to experience the future of AI automation.
                </p>
              </div>
              
              <div className="bg-background/50 p-6 rounded-lg border border-border/50 shadow-sm">
                <div className="flex items-center gap-2 mb-3">
                  <span className="p-1.5 rounded-full bg-emerald-500/10 text-emerald-500">
                    <Bell className="h-4 w-4" />
                  </span>
                  <h3 className="text-lg font-semibold">Launch Notification</h3>
                </div>
                <p className="text-muted-foreground">
                  Subscribe to receive launch updates and exclusive early-bird pricing offers.
                </p>
              </div>
            </div>
            
            {/* CTA - Early access signup */}
            <Link href="/platform">
              <Button size="lg" className="group relative overflow-hidden">
                <span className="relative z-10 flex items-center">
                  Learn More About Platform 2.0
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </span>
                <span className="absolute inset-0 bg-gradient-to-r from-brand via-purple-500 to-brand bg-[length:200%_100%] animate-rainbow opacity-80"></span>
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}