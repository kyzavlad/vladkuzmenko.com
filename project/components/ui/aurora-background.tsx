"use client";

import { cn } from "@/lib/utils";

export function AuroraBackground({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "relative min-h-screen flex flex-col items-center justify-center antialiased bg-black",
        className
      )}
    >
      <div
        className="absolute inset-0 -z-10 h-full w-full bg-black"
        style={{
          backgroundImage: `
            radial-gradient(circle at 50% 50%, rgba(var(--brand-rgb), 0.1), transparent 25%),
            radial-gradient(circle at 80% 20%, rgba(var(--brand-rgb), 0.1), transparent 35%)
          `,
          backgroundSize: "100% 100%, 150% 150%",
        }}
      >
        <div className="absolute inset-0 bg-black/20" />
      </div>
      <div
        className="absolute inset-0 -z-10 h-full w-full animate-aurora"
        style={{
          backgroundImage: `
            radial-gradient(circle at 50% 50%, rgba(var(--brand-rgb), 0.1), transparent 25%),
            radial-gradient(circle at 80% 20%, rgba(var(--brand-rgb), 0.1), transparent 35%)
          `,
          backgroundSize: "100% 100%, 150% 150%",
        }}
      />
      {children}
    </div>
  );
}