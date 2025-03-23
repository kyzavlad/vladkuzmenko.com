{`"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { buttonVariants } from "@/components/ui/button";

export type CalendarProps = React.HTMLAttributes<HTMLDivElement>;

export function Calendar({ className, ...props }: CalendarProps) {
  return (
    <div className={cn("p-3", className)} {...props}>
      <div className="space-y-4">
        {/* Simple calendar header */}
        <div className="flex items-center justify-between">
          <button className={cn(buttonVariants({ variant: "outline" }), "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100")}>
            <ChevronLeft className="h-4 w-4" />
          </button>
          <div className="font-medium">March 2025</div>
          <button className={cn(buttonVariants({ variant: "outline" }), "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100")}>
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        {/* Simple calendar grid */}
        <div className="grid grid-cols-7 gap-1 text-sm">
          {["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"].map((day) => (
            <div key={day} className="text-center text-muted-foreground">
              {day}
            </div>
          ))}
          {Array.from({ length: 31 }, (_, i) => (
            <button
              key={i + 1}
              className={cn(
                "aspect-square p-2 text-center text-sm rounded-md hover:bg-accent",
                "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              )}
            >
              {i + 1}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}`}