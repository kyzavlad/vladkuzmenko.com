"use client";

import { cn } from "@/lib/utils";
import { 
  Terminal, 
  Sliders, 
  DollarSign, 
  Cloud, 
  Route, 
  HelpCircle, 
  Settings, 
  Heart 
} from "lucide-react";

export function FeaturesSectionWithHoverEffects() {
  const features = [
    {
      title: "Automated Support Service",
      description: "AI-powered chatbots and voice assistants that handle inquiries, provide consultations, and answer technical questions 24/7.",
      icon: <HelpCircle size={24} />,
    },
    {
      title: "Personalized Experience",
      description: "Our AI learns from each interaction to deliver increasingly personalized customer experiences.",
      icon: <Sliders size={24} />,
    },
    {
      title: "Cost-Effective Solution",
      description: "Reduce support costs by up to 70% while improving customer satisfaction and response times.",
      icon: <DollarSign size={24} />,
    },
    {
      title: "Cloud-Based Infrastructure",
      description: "Secure, scalable cloud infrastructure with 99.9% uptime guarantee for reliable service.",
      icon: <Cloud size={24} />,
    },
    {
      title: "Seamless Integration",
      description: "Easily integrate with your existing CRM, email marketing, and customer service platforms.",
      icon: <Route size={24} />,
    },
    {
      title: "24/7 Technical Support",
      description: "Our team of experts is available around the clock to ensure your AI automation runs smoothly.",
      icon: <Terminal size={24} />,
    },
    {
      title: "Customizable Workflows",
      description: "Tailor automation workflows to match your specific business processes and requirements.",
      icon: <Settings size={24} />,
    },
    {
      title: "Customer-Centric Design",
      description: "Built with your customers in mind, our solutions create delightful experiences that build loyalty.",
      icon: <Heart size={24} />,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 relative z-10 py-10 max-w-7xl mx-auto">
      {features.map((feature, index) => (
        <Feature key={feature.title} {...feature} index={index} />
      ))}
    </div>
  );
}

const Feature = ({
  title,
  description,
  icon,
  index,
}: {
  title: string;
  description: string;
  icon: React.ReactNode;
  index: number;
}) => {
  return (
    <div
      className={cn(
        "flex flex-col lg:border-r py-10 relative group/feature dark:border-neutral-800",
        (index === 0 || index === 4) && "lg:border-l dark:border-neutral-800",
        index < 4 && "lg:border-b dark:border-neutral-800"
      )}
    >
      {index < 4 && (
        <div className="opacity-0 group-hover/feature:opacity-100 transition duration-200 absolute inset-0 h-full w-full bg-gradient-to-t from-neutral-100 dark:from-neutral-800 to-transparent pointer-events-none" />
      )}
      {index >= 4 && (
        <div className="opacity-0 group-hover/feature:opacity-100 transition duration-200 absolute inset-0 h-full w-full bg-gradient-to-b from-neutral-100 dark:from-neutral-800 to-transparent pointer-events-none" />
      )}
      <div className="mb-4 relative z-10 px-10 text-neutral-600 dark:text-neutral-400">
        {icon}
      </div>
      <div className="text-lg font-bold mb-2 relative z-10 px-10">
        <div className="absolute left-0 inset-y-0 h-6 group-hover/feature:h-8 w-1 rounded-tr-full rounded-br-full bg-neutral-300 dark:bg-neutral-700 group-hover/feature:bg-brand transition-all duration-200 origin-center" />
        <span className="group-hover/feature:translate-x-2 transition duration-200 inline-block text-neutral-800 dark:text-neutral-100">
          {title}
        </span>
      </div>
      <p className="text-sm text-neutral-600 dark:text-neutral-300 max-w-xs relative z-10 px-10">
        {description}
      </p>
    </div>
  );
};