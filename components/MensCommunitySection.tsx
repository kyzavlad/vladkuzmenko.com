"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ArrowRight, Shield, Star, Zap } from "lucide-react";
import Link from "next/link";

export function MensCommunitySection() {
  return (
    <div className="w-full py-16 md:py-20 bg-background">
      <div className="container mx-auto">
        <div className="max-w-5xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h2 className="text-4xl md:text-6xl font-bold mb-6 text-gradient">
              Warriors Team
            </h2>
            <p className="text-2xl md:text-3xl font-semibold mb-4">
              99.9% of modern-day men will never experience the power of true Brotherhood and Elite Community.
            </p>
            <p className="text-xl text-muted-foreground">
              They will never know what it's like to be surrounded by ambitious, driven men who push each other to greatness.
              To be part of a brotherhood that demands excellence and accepts nothing less.
            </p>
          </div>

          {/* Content Grid */}
          <div className="grid md:grid-cols-2 gap-12">
            {/* Left Column - Text */}
            <div className="space-y-8">
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <Shield className="h-6 w-6 text-purple-500" />
                  <h3 className="text-2xl font-bold">Elite Standards</h3>
                </div>
                <p className="text-muted-foreground text-lg">
                  We are building a community of exceptional men who refuse to settle for mediocrity. 
                  Men who understand that greatness requires discipline, dedication, and unwavering commitment.
                </p>
              </div>

              <div>
                <div className="flex items-center gap-3 mb-4">
                  <Star className="h-6 w-6 text-blue-500" />
                  <h3 className="text-2xl font-bold">Unmatched Support</h3>
                </div>
                <p className="text-muted-foreground text-lg">
                  In our community, you'll find mentors, allies, and brothers who understand your journey. 
                  Men who will celebrate your victories and help you learn from your setbacks.
                </p>
              </div>

              <div>
                <div className="flex items-center gap-3 mb-4">
                  <Zap className="h-6 w-6 text-amber-500" />
                  <h3 className="text-2xl font-bold">Transformative Growth</h3>
                </div>
                <p className="text-muted-foreground text-lg">
                  Every member of our community is committed to constant improvement - in business, 
                  relationships, health, and mindset. We push each other to break through limitations.
                </p>
              </div>
            </div>

            {/* Right Column - Images */}
            <div className="space-y-6">
              <div className="w-full aspect-video rounded-lg overflow-hidden">
                <img
                  src="/Снимок-экрана-2025-03-05-в-11.43.38 (1).webp"
                  alt="Warriors Team Meeting"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="w-full aspect-video rounded-lg overflow-hidden">
                <img
                  src="/Снимок-экрана-2025-03-05-в-11.44.15 (1).webp"
                  alt="Team Training"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="w-full aspect-video rounded-lg overflow-hidden">
                <img
                  src="/Снимок-экрана-2025-03-05-в-11.44.47 (1).webp"
                  alt="Team Success"
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          </div>

          {/* CTA */}
          <div className="text-center mt-12">
            <h3 className="text-2xl md:text-3xl font-bold mb-6">
              The World Needs Men Like You
            </h3>
            <p className="text-muted-foreground mb-8 max-w-2xl mx-auto text-lg">
              If you have the drive, the ambition, and the commitment to excellence, 
              you belong here. Join a brotherhood of men who are redefining what's possible.
            </p>
            <Link href="/team">
              <Button 
                size="lg"
                className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white group"
              >
                Join The Warriors Team
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}