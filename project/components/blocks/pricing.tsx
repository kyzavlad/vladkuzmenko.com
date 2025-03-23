"use client";

import { buttonVariants } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useMediaQuery } from "@/hooks/use-media-query";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { Check, Star, Flame } from "lucide-react";
import Link from "next/link";
import { useState, useRef } from "react";

interface PricingPlan {
  name: string;
  price: string;
  yearlyPrice: string;
  period: string;
  features: string[];
  description: string;
  buttonText: string;
  href: string;
  isPopular: boolean;
  highlighted?: boolean;
  component?: React.ReactNode;
}

interface PricingProps {
  plans: PricingPlan[];
  title?: string;
  description?: string;
}

export function Pricing({
  plans,
  title = "Simple, Transparent Pricing",
  description = "Choose the plan that works for you All plans include access to our platform, lead generation tools, and dedicated support.",
}: PricingProps) {
  const [isMonthly, setIsMonthly] = useState(true);
  const isDesktop = useMediaQuery("(min-width: 768px)");
  const switchRef = useRef<HTMLButtonElement>(null);

  const handleToggle = (checked: boolean) => {
    setIsMonthly(!checked);
    if (checked && switchRef.current) {
      const rect = switchRef.current.getBoundingClientRect();
      const x = rect.left + rect.width / 2;
      const y = rect.top + rect.height / 2;

      // Simplified confetti effect without the external library
      const colors = [
        "hsl(var(--primary))",
        "hsl(var(--accent))",
        "hsl(var(--secondary))",
        "hsl(var(--muted))",
      ];
      
      // Create a simple visual feedback instead
      const feedbackEl = document.createElement('div');
      feedbackEl.textContent = '🎉';
      feedbackEl.style.position = 'absolute';
      feedbackEl.style.left = `${x}px`;
      feedbackEl.style.top = `${y}px`;
      feedbackEl.style.fontSize = '24px';
      feedbackEl.style.transform = 'translate(-50%, -50%)';
      feedbackEl.style.zIndex = '9999';
      feedbackEl.style.pointerEvents = 'none';
      document.body.appendChild(feedbackEl);
      
      // Animate and remove
      setTimeout(() => {
        feedbackEl.style.transition = 'all 0.5s ease-out';
        feedbackEl.style.opacity = '0';
        feedbackEl.style.transform = 'translate(-50%, -100px)';
        setTimeout(() => {
          document.body.removeChild(feedbackEl);
        }, 500);
      }, 100);
    }
  };

  return (
    <div className="container py-20">
      <div className="text-center space-y-4 mb-12">
        <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
          {title}
        </h2>
        <p className="text-muted-foreground text-lg whitespace-pre-line">
          {description}
        </p>
      </div>

      <div className="flex justify-center mb-10">
        <label className="relative inline-flex items-center cursor-pointer">
          <Label>
            <Switch
              ref={switchRef as any}
              checked={!isMonthly}
              onCheckedChange={handleToggle}
              className="relative"
            />
          </Label>
        </label>
        <span className="ml-2 font-semibold">
          Annual billing <span className="text-primary">(Save 20%)</span>
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {plans.map((plan, index) => (
          <motion.div
            key={index}
            initial={{ y: 50, opacity: 1 }}
            whileInView={
              isDesktop
                ? {
                    y: plan.isPopular ? -20 : 0,
                    opacity: 1,
                    scale: plan.isPopular ? 1.05 : 1.0,
                  }
                : {}
            }
            viewport={{ once: true }}
            transition={{
              duration: 1.6,
              type: "spring",
              stiffness: 100,
              damping: 30,
              delay: 0.4 + index * 0.1,
              opacity: { duration: 0.5 },
            }}
            className={cn(
              `rounded-2xl border-[1px] p-6 bg-background text-center lg:flex lg:flex-col lg:justify-center relative`,
              plan.isPopular ? "border-primary border-2" : "border-border",
              plan.highlighted ? "border-gold-medium border-2" : "",
              "flex flex-col",
              !plan.isPopular && !plan.highlighted && "mt-5",
              "z-10"
            )}
          >
            {plan.isPopular && (
              <div className="absolute top-0 right-0 bg-primary py-0.5 px-2 rounded-bl-xl rounded-tr-xl flex items-center">
                <Flame className="text-primary-foreground h-4 w-4 fill-current" />
                <span className="text-primary-foreground ml-1 font-sans font-semibold">
                  Most Popular
                </span>
              </div>
            )}
            {plan.highlighted && !plan.isPopular && (
              <div className="absolute top-0 right-0 bg-gradient-to-r from-gold-light to-gold-dark py-0.5 px-2 rounded-bl-xl rounded-tr-xl flex items-center">
                <span className="text-black ml-1 font-sans font-semibold">
                  Customizable
                </span>
              </div>
            )}
            <div className="flex-1 flex flex-col">
              <p className="text-base font-semibold text-muted-foreground">
                {plan.name}
              </p>
              <div className="mt-6 flex items-center justify-center gap-x-2">
                <span className="text-5xl font-bold tracking-tight text-foreground">
                  ${isMonthly ? plan.price : plan.yearlyPrice}
                </span>
              </div>
              <p className="text-sm font-semibold leading-6 tracking-wide text-muted-foreground">
                one-time
              </p>
              <ul className="mt-5 gap-2 flex flex-col">
                {plan.features.map((feature, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <Check className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                    <span className="text-left">{feature}</span>
                  </li>
                ))}
              </ul>
              <hr className="w-full my-4" />
              {plan.component ? (
                plan.component
              ) : (
                <Link
                  href={plan.href}
                  className={cn(
                    buttonVariants({
                      variant: "outline",
                    }),
                    "group relative w-full gap-2 overflow-hidden text-lg font-semibold tracking-tighter",
                    "transform-gpu ring-offset-current transition-all duration-300 ease-out hover:ring-2 hover:ring-primary hover:ring-offset-1 hover:bg-primary hover:text-primary-foreground",
                    plan.isPopular
                      ? "bg-primary text-primary-foreground"
                      : "bg-background text-foreground",
                    plan.highlighted
                      ? "bg-gradient-to-r from-gold-light to-gold-dark text-black hover:from-gold-medium hover:to-gold-dark"
                      : ""
                  )}
                >
                  {plan.buttonText}
                </Link>
              )}
              <p className="mt-6 text-xs leading-5 text-muted-foreground">
                {plan.description}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}