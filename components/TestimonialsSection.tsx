"use client";

import { TestimonialsSection as TestimonialsMarquee } from "@/components/blocks/testimonials-with-marquee";

export function TestimonialsSection() {
  const testimonials = [
    {
      author: {
        name: "Sarah Johnson",
        handle: "@sarahj_tech",
        avatar: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150&h=150&fit=crop&crop=face"
      },
      text: "The AI automation platform has transformed our customer service operations. We've reduced response times by 80% while improving customer satisfaction scores.",
      href: "https://twitter.com/sarahj_tech"
    },
    {
      author: {
        name: "Michael Chen",
        handle: "@mchen_ai",
        avatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face"
      },
      text: "Implementing the email automation system increased our open rates by 45% and conversion rates by 28%. The ROI has been incredible.",
      href: "https://twitter.com/mchen_ai"
    },
    {
      author: {
        name: "Elena Rodriguez",
        handle: "@elena_digital",
        avatar: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150&h=150&fit=crop&crop=face"
      },
      text: "The content generation AI has saved our marketing team countless hours. The quality of the output is consistently high and requires minimal editing."
    },
    {
      author: {
        name: "David Park",
        handle: "@dpark_tech",
        avatar: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&h=150&fit=crop&crop=face"
      },
      text: "We've deployed the AI automation platform across 12 countries, and the centralized management dashboard makes it easy to monitor performance globally.",
      href: "https://twitter.com/dpark_tech"
    },
    {
      author: {
        name: "Olivia Martinez",
        handle: "@olivia_m",
        avatar: "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?w=150&h=150&fit=crop&crop=face"
      },
      text: "The conversational AI has revolutionized how we handle customer inquiries. It understands context and provides human-like responses that our customers love."
    },
    {
      author: {
        name: "James Wilson",
        handle: "@jwilson_cto",
        avatar: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face"
      },
      text: "As a CTO, I appreciate the robust security features and compliance tools built into the platform. It's made global deployment much simpler.",
      href: "https://twitter.com/jwilson_cto"
    }
  ];

  return (
    <div id="testimonials-section" className="w-full bg-background">
      <TestimonialsMarquee
        title="Trusted by innovative companies worldwide"
        description="Join thousands of businesses that have transformed their operations with our AI automation platform"
        testimonials={testimonials}
      />
    </div>
  );
}