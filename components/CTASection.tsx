"use client";

import { RainbowButton } from "@/components/ui/rainbow-button";
import { ArrowRight } from "lucide-react";
import { ContactDialog } from "@/components/ui/contact-dialog";

export function CTASection() {
  return (
    <div id="cta-section" className="w-full py-20 md:py-32 bg-gradient-to-b from-background to-background/90">
      <div className="container mx-auto">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-5xl font-bold mb-6 text-gradient">
            Transform Your Business with AI Automation
          </h2>
          <p className="text-muted-foreground text-lg md:text-xl mb-10 max-w-2xl mx-auto">
            Join thousands of businesses that have revolutionized their operations with our AI-powered automation solutions. Get started today and see the difference.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <ContactDialog triggerText="Get Started Now" isRainbowButton={true} />
            
            <div className="text-sm text-muted-foreground">
              <p>No credit card required</p>
              <p>14-day free trial</p>
            </div>
          </div>
          
          <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
            <div>
              <p className="text-3xl md:text-4xl font-bold text-foreground">98%</p>
              <p className="text-sm text-muted-foreground">Customer Satisfaction</p>
            </div>
            <div>
              <p className="text-3xl md:text-4xl font-bold text-foreground">75%</p>
              <p className="text-sm text-muted-foreground">Cost Reduction</p>
            </div>
            <div>
              <p className="text-3xl md:text-4xl font-bold text-foreground">24/7</p>
              <p className="text-sm text-muted-foreground">Support Available</p>
            </div>
            <div>
              <p className="text-3xl md:text-4xl font-bold text-foreground">300+</p>
              <p className="text-sm text-muted-foreground">Active Clients</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}