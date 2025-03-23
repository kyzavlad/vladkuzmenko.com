"use client";

import { FeaturesSectionWithHoverEffects } from "@/components/blocks/feature-section-with-hover-effects";

export function FeaturesSection() {
  return (
    <div id="features-section" className="w-full py-16 md:py-24 bg-background">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gradient">Our AI Automation Features</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Discover how our AI-powered automation solutions can transform your business operations and enhance customer experiences.
          </p>
        </div>
        <FeaturesSectionWithHoverEffects />
      </div>
    </div>
  );
}