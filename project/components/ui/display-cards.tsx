"use client";

import { cn } from "@/lib/utils";
import { Sparkles } from "lucide-react";

interface DisplayCardProps {
  className?: string;
  icon?: React.ReactNode;
  title?: string;
  description?: string;
  date?: string;
  iconClassName?: string;
  titleClassName?: string;
  wrapper?: (cardInner: React.ReactNode) => React.ReactNode;
}

function DisplayCard({
  className,
  icon = <Sparkles className="size-4 text-blue-300" />,
  title = "Featured",
  description = "Discover amazing content",
  date = "Just now",
  iconClassName = "text-blue-500",
  titleClassName = "text-blue-500",
  wrapper
}: DisplayCardProps) {
  const cardContent = (
    <div
      className={cn(
        "relative flex h-[140px] w-[260px] -skew-y-[8deg] select-none flex-col justify-between rounded-xl border-2 bg-muted/70 backdrop-blur-sm p-4 transition-all duration-300",
        "hover:-translate-y-4 hover:bg-background/80 hover:border-brand/20",
        "after:absolute after:-right-1 after:top-[-5%] after:h-[110%] after:w-[20rem] after:bg-gradient-to-l after:from-background after:to-transparent after:content-['']",
        className
      )}
    >
      <div className="flex items-center gap-2">
        <span className="relative inline-block rounded-full bg-blue-800/30 p-1.5">
          {icon}
        </span>
        <p className={cn("text-base font-medium line-clamp-1", titleClassName)}>{title}</p>
      </div>
      <p className="text-sm line-clamp-2">{description}</p>
      <p className="text-xs text-muted-foreground">{date}</p>
    </div>
  );

  return wrapper ? wrapper(cardContent) : cardContent;
}

interface DisplayCardsProps {
  cards?: DisplayCardProps[];
}

export default function DisplayCards({ cards }: DisplayCardsProps) {
  const defaultCards = [
    {
      className: "[grid-area:stack] before:absolute before:w-[100%] before:outline-1 before:rounded-xl before:outline-border before:h-[100%] before:content-[''] before:bg-blend-overlay before:bg-background/50 grayscale-[100%] hover:before:opacity-0 before:transition-opacity before:duration:700 hover:grayscale-0 before:left-0 before:top-0",
    },
    {
      className: "[grid-area:stack] translate-x-[160px] before:absolute before:w-[100%] before:outline-1 before:rounded-xl before:outline-border before:h-[100%] before:content-[''] before:bg-blend-overlay before:bg-background/50 grayscale-[100%] hover:before:opacity-0 before:transition-opacity before:duration:700 hover:grayscale-0 before:left-0 before:top-0",
    },
    {
      className: "[grid-area:stack] translate-x-[320px] before:absolute before:w-[100%] before:outline-1 before:rounded-xl before:outline-border before:h-[100%] before:content-[''] before:bg-blend-overlay before:bg-background/50 grayscale-[100%] hover:before:opacity-0 before:transition-opacity before:duration:700 hover:grayscale-0 before:left-0 before:top-0",
    },
    {
      className: "[grid-area:stack] translate-x-[480px] before:absolute before:w-[100%] before:outline-1 before:rounded-xl before:outline-border before:h-[100%] before:content-[''] before:bg-blend-overlay before:bg-background/50 grayscale-[100%] hover:before:opacity-0 before:transition-opacity before:duration:700 hover:grayscale-0 before:left-0 before:top-0",
    },
    {
      className: "[grid-area:stack] translate-x-[640px] before:absolute before:w-[100%] before:outline-1 before:rounded-xl before:outline-border before:h-[100%] before:content-[''] before:bg-blend-overlay before:bg-background/50 grayscale-[100%] hover:before:opacity-0 before:transition-opacity before:duration:700 hover:grayscale-0 before:left-0 before:top-0",
    }
  ];

  const displayCards = cards || defaultCards;

  return (
    <div className="relative w-full h-[180px] overflow-x-auto overflow-y-visible pb-4 md:overflow-visible">
      <div className="absolute left-1/2 -translate-x-1/2 flex -space-x-40 md:-space-x-32">
        {displayCards.map((cardProps, index) => (
          <DisplayCard key={index} {...cardProps} />
        ))}
      </div>
    </div>
  );
}