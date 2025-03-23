"use client";

import { createContext, useContext, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface SidebarContextValue {
  expanded: boolean;
  setExpanded: (expanded: boolean) => void;
}

const SidebarContext = createContext<SidebarContextValue>({
  expanded: true,
  setExpanded: () => {},
});

interface SidebarProps {
  children: React.ReactNode;
  className?: string;
}

export function Sidebar({ children, className }: SidebarProps) {
  const [expanded, setExpanded] = useState(true);

  return (
    <SidebarContext.Provider value={{ expanded, setExpanded }}>
      <motion.div
        className={cn(
          "h-full bg-background border-r border-border/40",
          className
        )}
      >
        {children}
      </motion.div>
    </SidebarContext.Provider>
  );
}

interface SidebarBodyProps {
  children: React.ReactNode;
  className?: string;
}

export function SidebarBody({ children, className }: SidebarBodyProps) {
  return (
    <motion.div className={cn("p-3 flex flex-col gap-1", className)}>
      {children}
    </motion.div>
  );
}

interface SidebarLinkProps {
  icon?: React.ReactNode;
  label: string;
  href?: string;
  onClick?: () => void;
  active?: boolean;
  className?: string;
}

export function SidebarLink({ 
  icon, 
  label, 
  href = "#",
  onClick,
  active,
  className 
}: SidebarLinkProps) {
  const { expanded } = useContext(SidebarContext);
  const pathname = usePathname();
  const isActive = active || pathname === href;

  const content = (
    <div
      className={cn(
        "flex items-center gap-4 px-3 py-2 rounded-lg transition-colors",
        "hover:bg-accent/50 cursor-pointer",
        isActive && "bg-accent text-accent-foreground",
        className
      )}
      onClick={onClick}
    >
      {icon && <span className="w-6 h-6">{icon}</span>}
      <span className="text-sm font-medium">{label}</span>
    </div>
  );

  if (onClick) {
    return content;
  }

  return <Link href={href}>{content}</Link>;
}