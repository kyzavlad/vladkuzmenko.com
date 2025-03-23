"use client";

import { WorldMap } from "@/components/ui/world-map";
import { motion } from "framer-motion";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { Sun, Moon } from "lucide-react";
import { useState, useEffect } from "react";

export function MapSection() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  
  // Only show theme toggle after component is mounted to avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div id="map-section" className="w-full py-16 md:py-24 bg-background">
      <div className="container mx-auto">
        <div className="max-w-7xl mx-auto text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gradient">
            Global AI Automation Network
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Our AI automation solutions connect businesses worldwide, enabling seamless operations across borders and time zones.
          </p>
          <div className="mt-6">
            <p className="font-bold text-xl md:text-2xl text-foreground">
              Worldwide{" "}
              <span className="text-neutral-400">
                {"Connectivity".split("").map((word, idx) => (
                  <motion.span
                    key={idx}
                    className="inline-block"
                    initial={{ x: -10, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ duration: 0.5, delay: idx * 0.04 }}
                  >
                    {word}
                  </motion.span>
                ))}
              </span>
            </p>
            <p className="text-sm md:text-lg text-muted-foreground max-w-2xl mx-auto py-4">
              Break free from traditional boundaries. Deploy AI automation solutions anywhere, with centralized management and monitoring. Perfect for global enterprises and distributed teams.
            </p>
            
            {/* Theme toggle button */}
            {mounted && (
              <div className="flex justify-center mb-6">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                  className="rounded-full p-2 h-10 w-10"
                >
                  {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                  <span className="sr-only">Toggle theme</span>
                </Button>
              </div>
            )}
          </div>
        </div>

        <div className="max-w-6xl mx-auto">
          <WorldMap
            dots={[
              {
                start: { lat: 40.7128, lng: -74.006 }, // New York
                end: { lat: 51.5074, lng: -0.1278 }, // London
              },
              {
                start: { lat: 51.5074, lng: -0.1278 }, // London
                end: { lat: 48.8566, lng: 2.3522 }, // Paris
              },
              {
                start: { lat: 48.8566, lng: 2.3522 }, // Paris
                end: { lat: 55.7558, lng: 37.6173 }, // Moscow
              },
              {
                start: { lat: 55.7558, lng: 37.6173 }, // Moscow
                end: { lat: 39.9042, lng: 116.4074 }, // Beijing
              },
              {
                start: { lat: 39.9042, lng: 116.4074 }, // Beijing
                end: { lat: 35.6762, lng: 139.6503 }, // Tokyo
              },
              {
                start: { lat: 35.6762, lng: 139.6503 }, // Tokyo
                end: { lat: -33.8688, lng: 151.2093 }, // Sydney
              },
              {
                start: { lat: -33.8688, lng: 151.2093 }, // Sydney
                end: { lat: -33.9249, lng: 18.4241 }, // Cape Town
              },
              {
                start: { lat: -33.9249, lng: 18.4241 }, // Cape Town
                end: { lat: 19.4326, lng: -99.1332 }, // Mexico City
              },
              {
                start: { lat: 19.4326, lng: -99.1332 }, // Mexico City
                end: { lat: 37.7749, lng: -122.4194 }, // San Francisco
              },
              {
                start: { lat: 37.7749, lng: -122.4194 }, // San Francisco
                end: { lat: 40.7128, lng: -74.006 }, // New York
              },
            ]}
            lineColor={theme === "dark" ? "#3b82f6" : "#2563eb"}
          />
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div className="bg-background/50 p-6 rounded-lg border border-border/50 shadow-sm">
            <h3 className="text-xl font-semibold mb-3">Global Deployment</h3>
            <p className="text-muted-foreground">
              Deploy AI automation solutions in over 120 countries with localized language support and regional compliance features.
            </p>
          </div>
          <div className="bg-background/50 p-6 rounded-lg border border-border/50 shadow-sm">
            <h3 className="text-xl font-semibold mb-3">Centralized Management</h3>
            <p className="text-muted-foreground">
              Monitor and manage all your AI automation workflows from a single dashboard, regardless of geographic location.
            </p>
          </div>
          <div className="bg-background/50 p-6 rounded-lg border border-border/50 shadow-sm">
            <h3 className="text-xl font-semibold mb-3">24/7 Global Support</h3>
            <p className="text-muted-foreground">
              Access technical support in any time zone with our distributed support team and AI-powered assistance.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}