"use client";

import React, { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface GoldenSparksProps {
  className?: string;
  intensity?: "low" | "medium" | "high";
  size?: "small" | "medium" | "large";
  duration?: number;
}

export function GoldenSparks({
  className,
  intensity = "medium",
  size = "medium",
  duration = 3,
}: GoldenSparksProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  
  const intensityMap = {
    low: 50,
    medium: 100,
    high: 200,
  };
  
  const sizeMap = {
    small: 2,
    medium: 3,
    large: 5,
  };

  class Particle {
    x: number;
    y: number;
    size: number;
    speedX: number;
    speedY: number;
    life: number;
    maxLife: number;
    color: string;
    rotation: number;
    rotationSpeed: number;

    constructor(x: number, y: number, size: number) {
      this.x = x;
      this.y = y;
      this.size = Math.random() * size + 1;
      
      // Random direction with more upward/outward tendency
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 3 + 1;
      
      this.speedX = Math.cos(angle) * speed;
      this.speedY = Math.sin(angle) * speed;
      
      // Life duration
      this.maxLife = Math.random() * 100 + 50;
      this.life = this.maxLife;
      
      // Golden colors
      const hue = Math.random() * 30 + 40; // Gold range
      const saturation = Math.random() * 20 + 80; // High saturation
      const lightness = Math.random() * 20 + 60; // Bright
      this.color = `hsla(${hue}, ${saturation}%, ${lightness}%, 1)`;
      
      // Rotation for sparkle effect
      this.rotation = Math.random() * Math.PI * 2;
      this.rotationSpeed = (Math.random() - 0.5) * 0.2;
    }

    update() {
      this.x += this.speedX;
      this.y += this.speedY;
      
      // Add slight gravity effect
      this.speedY += 0.03;
      
      // Slow down over time
      this.speedX *= 0.99;
      this.speedY *= 0.99;
      
      // Decrease life
      this.life--;
      
      // Rotate
      this.rotation += this.rotationSpeed;
      
      return this.life > 0;
    }

    draw(ctx: CanvasRenderingContext2D) {
      const opacity = this.life / this.maxLife;
      ctx.save();
      
      // Set composite operation for glow effect
      ctx.globalCompositeOperation = "lighter";
      
      // Translate to particle position
      ctx.translate(this.x, this.y);
      ctx.rotate(this.rotation);
      
      // Draw sparkle
      ctx.beginPath();
      
      // Main circle
      ctx.fillStyle = this.color.replace("1)", `${opacity})`);
      ctx.arc(0, 0, this.size, 0, Math.PI * 2);
      ctx.fill();
      
      // Cross shape for sparkle
      const sparkSize = this.size * 2;
      ctx.fillStyle = this.color.replace("1)", `${opacity * 0.7})`);
      
      // Horizontal line
      ctx.fillRect(-sparkSize / 2, -this.size / 4, sparkSize, this.size / 2);
      
      // Vertical line
      ctx.fillRect(-this.size / 4, -sparkSize / 2, this.size / 2, sparkSize);
      
      // Add glow
      const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, this.size * 2);
      gradient.addColorStop(0, this.color.replace("1)", `${opacity * 0.5})`));
      gradient.addColorStop(1, this.color.replace("1)", "0)"));
      
      ctx.beginPath();
      ctx.fillStyle = gradient;
      ctx.arc(0, 0, this.size * 2, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.restore();
    }
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas dimensions
    const resizeCanvas = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Center point for particle emission
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    // Animation loop
    let animationId: number;
    const animate = () => {
      // Clear with slight fade for trail effect
      ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Add new particles
      if (Math.random() < 0.3) {
        const burstCount = Math.floor(Math.random() * 5) + 1;
        for (let i = 0; i < burstCount; i++) {
          // Create particles near the center with some randomness
          const offsetX = (Math.random() - 0.5) * 100;
          const offsetY = (Math.random() - 0.5) * 100;
          particlesRef.current.push(
            new Particle(
              centerX + offsetX, 
              centerY + offsetY, 
              sizeMap[size]
            )
          );
        }
      }

      // Limit total particles
      if (particlesRef.current.length > intensityMap[intensity]) {
        particlesRef.current = particlesRef.current.slice(-intensityMap[intensity]);
      }

      // Update and draw particles
      particlesRef.current = particlesRef.current.filter(particle => {
        const isAlive = particle.update();
        if (isAlive) {
          particle.draw(ctx);
        }
        return isAlive;
      });

      animationId = requestAnimationFrame(animate);
    };

    animate();

    // Cleanup
    return () => {
      window.removeEventListener("resize", resizeCanvas);
      cancelAnimationFrame(animationId);
    };
  }, [intensity, size, duration]);

  return (
    <canvas
      ref={canvasRef}
      className={cn("absolute inset-0 z-0 w-full h-full", className)}
    />
  );
}