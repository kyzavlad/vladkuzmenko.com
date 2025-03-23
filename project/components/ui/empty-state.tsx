import { cn } from "@/lib/utils";
import { DivideIcon as LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

interface EmptyStateProps {
  title: string;
  description: string;
  icons?: Array<typeof LucideIcon>;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function EmptyState({
  title,
  description,
  icons = [],
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center p-8 rounded-lg border border-border/50",
        className
      )}
    >
      {icons.length > 0 && (
        <div className="flex -space-x-4 mb-4">
          {icons.map((Icon, index) => (
            <div
              key={index}
              className="w-12 h-12 rounded-full bg-muted flex items-center justify-center"
              style={{
                transform: `translateX(${index * 10}px)`,
                zIndex: icons.length - index,
              }}
            >
              <Icon className="w-6 h-6 text-muted-foreground" />
            </div>
          ))}
        </div>
      )}
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground mb-4">{description}</p>
      {action && (
        <Button onClick={action.onClick} variant="outline" size="sm">
          {action.label}
        </Button>
      )}
    </div>
  );
}