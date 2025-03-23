"use client";

import { Pricing } from "@/components/blocks/pricing";
import { PaymentDialog } from "@/components/ui/payment-dialog";
import { ContactDialog } from "@/components/ui/contact-dialog";

export function PricingSection() {
  const aiAutomationPlans = [
    {
      name: "STARTER",
      price: "2899",
      yearlyPrice: "2319",
      period: "one-time",
      features: [
        "AI Customer Support Bot (Basic)",
        "Email Automation (5 templates)",
        "Content Generation (10 pieces/month)",
        "Analytics Dashboard (Basic)",
        "1 Automation Workflow",
        "1,000 AI Interactions/month",
        "Email Support (48h response)",
        "Community Access"
      ],
      description: "Perfect for small businesses starting with AI automation",
      buttonText: "Get Started",
      href: "/sign-up",
      isPopular: false,
      component: <PaymentDialog 
                  planName="Starter Plan" 
                  planPrice="2899" 
                  planPeriod="one-time" 
                  buttonText="Get Started" 
                  buttonVariant="default" 
                  fullWidth={true} 
                />
    },
    {
      name: "PROFESSIONAL",
      price: "7499",
      yearlyPrice: "5999",
      period: "one-time",
      features: [
        "AI Customer Support Bot (Advanced)",
        "Email Automation (Unlimited templates)",
        "Content Generation (50 pieces/month)",
        "Advanced Analytics & Reporting",
        "25 Automation Workflows",
        "10,000 AI Interactions/month",
        "Priority Support (24h response)",
        "API Access",
        "Team Collaboration Tools",
        "Custom Integrations (2)"
      ],
      description: "Ideal for growing businesses with complex automation needs",
      buttonText: "Get Started",
      href: "/sign-up",
      isPopular: false,
      component: <PaymentDialog 
                  planName="Professional Plan" 
                  planPrice="7499" 
                  planPeriod="one-time" 
                  buttonText="Get Started" 
                  buttonVariant="default" 
                  fullWidth={true} 
                />
    },
    {
      name: "ENTERPRISE",
      price: "14599",
      yearlyPrice: "11679",
      period: "one-time",
      features: [
        "AI Customer Support Bot (Enterprise)",
        "Email Automation (Unlimited + Custom)",
        "Content Generation (Unlimited)",
        "Enterprise Analytics Suite",
        "Unlimited Automation Workflows",
        "Unlimited AI Interactions",
        "Dedicated Account Manager",
        "24/7 Premium Support",
        "Custom Integrations (Unlimited)",
        "Advanced Security Features",
        "SLA Guarantees",
        "Team Training & Onboarding",
        "Full Resource Access",
        "Custom AI Model Training"
      ],
      description: "For organizations requiring custom AI automation solutions",
      buttonText: "Contact Sales",
      href: "/contact",
      isPopular: false,
      component: <ContactDialog 
                  triggerText="Contact Sales" 
                  className="w-full" 
                />
    },
    {
      name: "CUSTOM",
      price: "899",
      yearlyPrice: "719",
      period: "one-time",
      features: [
        "Select only the services you need",
        "Flexible AI Support Bot options",
        "Pay-per-use Email Automation",
        "On-demand Content Generation",
        "Custom Analytics Dashboard",
        "Scalable Workflow options",
        "Pay-as-you-go AI Interactions",
        "Standard Support included",
        "Integration assistance",
        "Personalized pricing"
      ],
      description: "Tailored solution for specific automation needs",
      buttonText: "Get Custom Quote",
      href: "/custom-quote",
      isPopular: true,
      highlighted: true,
      component: <ContactDialog 
                  triggerText="Get Custom Quote" 
                  className="w-full" 
                />
    }
  ];

  return (
    <div id="pricing-section" className="w-full py-16 md:py-24 bg-background">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gradient">AI Automation Pricing</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Choose the perfect AI automation plan for your business needs. All plans include our core platform features with varying levels of AI capabilities and support.
          </p>
        </div>
        <Pricing 
          plans={aiAutomationPlans} 
          title="Transparent, Value-Based Pricing"
          description="Select the plan that aligns with your automation goals. All plans include our core AI platform, regular updates, and dedicated support."
        />
      </div>
    </div>
  );
}