"use client";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Mockup } from "@/components/ui/mockup";
import { Icons } from "@/components/ui/icons";
import { RainbowButton } from "@/components/ui/rainbow-button";
import { useState } from "react";
import { HeroVideoDialog } from "@/components/ui/hero-video-dialog";

interface HeroWithMockupProps {
  title: string;
  description: string;
  primaryCta?: {
    text: string;
    href: string;
    component?: React.ReactNode;
  };
  secondaryCta?: {
    text: string;
    href: string;
    icon?: React.ReactNode;
  };
  mockupImage?: {
    src: string;
    alt: string;
    width: number;
    height: number;
  };
  youtubeVideo?: {
    id: string;
    title: string;
    thumbnailSrc?: string;
  };
  className?: string;
}

export function HeroWithMockup({
  title,
  description,
  primaryCta = {
    text: "Get Started",
    href: "/get-started",
  },
  secondaryCta = {
    text: "GitHub",
    href: "https://github.com/your-repo",
    icon: <Icons.gitHub className="mr-2 h-4 w-4" />,
  },
  mockupImage,
  youtubeVideo,
  className,
}: HeroWithMockupProps) {
  const [isVideoLoading, setIsVideoLoading] = useState(true);

  return (
    <section
      className={cn(
        "relative bg-background text-foreground",
        "py-12 px-4 md:py-24 lg:py-32",
        "overflow-hidden",
        className,
      )}
    >
      <div className="relative mx-auto max-w-[1280px] flex flex-col gap-12 lg:gap-24">
        <div className="relative z-10 flex flex-col items-center gap-6 pt-8 md:pt-16 text-center lg:gap-12">
          {/* Heading */}
          <h1
            className={cn(
              "inline-block animate-appear",
              "bg-gradient-to-b from-foreground via-foreground/90 to-muted-foreground",
              "bg-clip-text text-transparent",
              "text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl",
              "leading-[1.1] sm:leading-[1.1]",
              "drop-shadow-sm dark:drop-shadow-[0_0_15px_rgba(255,255,255,0.1)]",
            )}
          >
            {title}
          </h1>
          {/* Description */}
          <p
            className={cn(
              "max-w-[550px] animate-appear opacity-0 [animation-delay:150ms]",
              "text-base sm:text-lg md:text-xl",
              "text-muted-foreground",
              "font-medium",
            )}
          >
            {description}
          </p>
          {/* CTAs */}
          <div className="relative z-10 flex flex-wrap justify-center gap-4 animate-appear opacity-0 [animation-delay:300ms]">
            {primaryCta.component ? (
              primaryCta.component
            ) : (
              <RainbowButton onClick={() => window.location.href = primaryCta.href}>
                {primaryCta.text}
              </RainbowButton>
            )}
            <Button
              asChild
              size="lg"
              variant="ghost"
              className={cn(
                "text-foreground/80 dark:text-foreground/70",
                "transition-all duration-300",
              )}
            >
              <a href={secondaryCta.href}>
                {secondaryCta.icon}
                {secondaryCta.text}
              </a>
            </Button>
          </div>
          {/* Mockup or YouTube Video */}
          <div className="relative w-full pt-12 px-4 sm:px-6 lg:px-8">
            <Mockup
              className={cn(
                "animate-appear opacity-0 [animation-delay:700ms]",
                "shadow-[0_0_50px_-12px_rgba(0,0,0,0.3)] dark:shadow-[0_0_50px_-12px_rgba(255,255,255,0.1)]",
                "border-brand/10 dark:border-brand/5",
              )}
            >
              {youtubeVideo ? (
                youtubeVideo.thumbnailSrc ? (
                  <HeroVideoDialog
                    animationStyle="from-center"
                    videoSrc={`https://www.youtube-nocookie.com/embed/${youtubeVideo.id}?autoplay=1`}
                    thumbnailSrc={youtubeVideo.thumbnailSrc}
                    thumbnailAlt={youtubeVideo.title}
                  />
                ) : (
                  <div className="relative w-full pb-[56.25%] h-0">
                    {isVideoLoading && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/10">
                        <span className="loader-sm"></span>
                      </div>
                    )}
                    <iframe
                      src={`https://www.youtube-nocookie.com/embed/${youtubeVideo.id}`}
                      title={youtubeVideo.title}
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                      className="absolute top-0 left-0 w-full h-full"
                      onLoad={() => setIsVideoLoading(false)}
                    ></iframe>
                  </div>
                )
              ) : mockupImage ? (
                <img
                  {...mockupImage}
                  className="w-full h-auto"
                  loading="lazy"
                  decoding="async"
                />
              ) : null}
            </Mockup>
          </div>
        </div>
      </div>
    </section>
  );
}