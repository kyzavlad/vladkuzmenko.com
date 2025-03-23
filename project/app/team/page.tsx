"use client";

import { useState, useEffect, useCallback } from "react";
import { Header } from "@/components/ui/header";
import { motion } from "framer-motion";
import { FooterSection } from "@/components/FooterSection";
import { CampusDashboard } from "@/components/CampusDashboard";
import { AuroraBackground } from "@/components/ui/aurora-background";
import { useLocalStorage } from "usehooks-ts";
import { cn } from "@/lib/utils";

export default function TeamPage() {
  const [mounted, setMounted] = useState(false);
  const [hasAccess] = useLocalStorage("platform-access", false);

  const initializeComponent = useCallback(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    initializeComponent();
  }, [initializeComponent]);

  // Only render the platform content after the component has mounted
  // This ensures hydration matches between server and client
  if (!mounted) {
    return null;
  }

  return (
    <main className="min-h-screen bg-black">
      <Header />
      
      <AuroraBackground>
        <motion.div
          initial={{ opacity: 0.0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{
            delay: 0.3,
            duration: 0.8,
            ease: "easeInOut",
          }}
          className="relative flex flex-col gap-4 items-center justify-center min-h-[50vh] px-4 pt-20 pb-12"
        >
          <div className="text-4xl md:text-7xl font-bold dark:text-white">
            Warriors Team Platform
          </div>
          <div className="font-extralight text-base md:text-4xl dark:text-neutral-200 py-4 max-w-3xl text-center">
            Join an elite community of ambitious men dedicated to growth, success, and brotherhood.
          </div>
        </motion.div>
      </AuroraBackground>

      <section className="w-full p-6 md:p-8 relative">
        <div className={cn(
          "rounded-2xl border border-neutral-800 bg-[#1e1e1e] overflow-hidden",
          "transition-all duration-300 relative",
          "shadow-[0_0_50px_-12px_rgba(0,122,255,0.15)]",
          "dark:shadow-[0_0_50px_-12px_rgba(51,153,255,0.15)]"
        )}>
          <CampusDashboard />
        </div>
      </section>

      <FooterSection />
    </main>
  );
}