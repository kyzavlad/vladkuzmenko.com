"use client";

import { cn } from "@/lib/utils";
import React from "react";

interface GlowProps {
  variant?: "above" | "below";
  className?: string;
}

export function Glow({ variant = "above", className }: GlowProps) {
  return (
    <div
      className={cn(
        "absolute inset-0 z-0 overflow-hidden",
        variant === "above" ? "z-10" : "z-0",
        className
      )}
    >
      <div
        className={cn(
          "absolute top-1/2 left-1/2 h-[120%] w-[120%] -translate-x-1/2 -translate-y-1/2 rounded-full",
          "bg-gradient-radial from-brand/20 via-transparent to-transparent",
          "opacity-50 blur-3xl"
        )}
      />
    </div>
  );
}