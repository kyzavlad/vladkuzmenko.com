"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface TimelineProps {
  data: Array<{
    title: string;
    content: React.ReactNode;
  }>;
}

export function Timeline({ data }: TimelineProps) {
  return (
    <div className="w-full bg-background font-sans md:px-10">
      <div className="relative max-w-7xl mx-auto pb-20">
        {/* Timeline line */}
        <div className="absolute left-4 md:left-0 top-0 h-full w-0.5 bg-gradient-to-b from-brand/50 via-brand/20 to-transparent" />

        {data.map((item, index) => (
          <div key={index} className="flex justify-start pt-10 md:pt-40 md:gap-10">
            {/* Timeline dot */}
            <div className="relative pl-4 md:pl-0">
              <div className="absolute left-[0.5px] -translate-x-1/2 h-4 w-4">
                <div className="h-full w-full rounded-full bg-background border-2 border-brand" />
              </div>
            </div>

            {/* Content */}
            <div className="relative pl-20 pr-4 md:pl-4 w-full">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
              >
                <h3 className="text-4xl font-bold mb-8 text-gradient">{item.title}</h3>
                {item.content}
              </motion.div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}