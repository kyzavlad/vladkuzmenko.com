"use client";

import { ThreeDPhotoCarousel } from "@/components/ui/3d-carousel";

export function GallerySection() {
  return (
    <div className="w-full py-16 md:py-24 bg-background">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gradient">AI Automation Gallery</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Explore our interactive showcase of AI automation solutions in action. Drag to rotate the carousel and click on any image to view details.
          </p>
        </div>
        <div className="max-w-5xl mx-auto">
          <ThreeDPhotoCarousel />
        </div>
        <div className="mt-12 text-center">
          <p className="text-sm text-muted-foreground">
            These examples represent real-world applications of our AI automation technology across various industries.
          </p>
        </div>
      </div>
    </div>
  );
}