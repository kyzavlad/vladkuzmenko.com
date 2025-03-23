"use client";

import { Timeline } from "@/components/ui/timeline";
import { RainbowButton } from "@/components/ui/rainbow-button";
import { motion, useScroll, useTransform } from "framer-motion";
import { Trophy } from "lucide-react";
import { useRef } from "react";
import Image from "next/image";
import { cn } from "@/lib/utils";

export function ProjectsSection() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start end", "end start"],
  });

  const data = [
    {
      id: "gorilla-mind",
      title: "Gorilla Mind",
      content: (
        <div className="flex flex-col gap-8">
          <div className="w-full prose prose-lg dark:prose-invert">
            <p className="text-lg text-foreground/90 leading-relaxed">
              We developed a comprehensive AI-powered customer support automation system for Gorilla Mind, 
              a leading e-commerce platform specializing in premium sports supplements.
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>Increased customer retention by 45%</li>
              <li>Boosted average order value by 28% through AI-powered cross-selling</li>
              <li>Reduced response time by 90% with 24/7 automated support</li>
              <li>Achieved 95% customer satisfaction rate</li>
            </ul>
            <blockquote className="border-l-4 border-brand pl-4 italic mt-4">
              "The AI automation system has revolutionized our customer support and sales operations. The intelligent product recommendations and 24/7 support have significantly improved our customer experience."
              <footer className="text-sm mt-2">- Derek Thompson, Director of E-commerce</footer>
            </blockquote>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <ProjectImage src="https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=800&h=450&fit=crop" alt="Gorilla Mind platform" />
            <ProjectImage src="https://images.unsplash.com/photo-1579758629938-03607ccdbaba?w=800&h=450&fit=crop" alt="Sports supplements" />
            <ProjectImage src="https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04?w=800&h=450&fit=crop" alt="Supplement manufacturing" />
          </div>
        </div>
      ),
    },
    {
      title: "LuxStay Hotels",
      content: (
        <div className="flex flex-col gap-8">
          <div className="w-full prose prose-lg dark:prose-invert">
            <p className="text-lg text-foreground/90 leading-relaxed">
              Transformed guest experience across LuxStay's international chain of luxury hotels through comprehensive AI automation.
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>Increased booking conversion rate by 45%</li>
              <li>Reduced check-in time by 70% with AI automation</li>
              <li>90% guest queries resolved by AI within 2 minutes</li>
              <li>35% increase in ancillary revenue through AI recommendations</li>
            </ul>
            <blockquote className="border-l-4 border-brand pl-4 italic mt-4">
              "The AI system has revolutionized how we serve our guests. From booking to checkout, every interaction is seamless and personalized."
              <footer className="text-sm mt-2">- James Morrison, CEO of LuxStay Hotels</footer>
            </blockquote>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <ProjectImage src="https://images.unsplash.com/photo-1566073771259-6a8506099945?w=800&h=450&fit=crop" alt="LuxStay Hotel lobby" />
            <ProjectImage src="https://images.unsplash.com/photo-1582719508461-905c673771fd?w=800&h=450&fit=crop" alt="Luxury hotel room" />
            <ProjectImage src="https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=800&h=450&fit=crop" alt="Hotel exterior" />
          </div>
        </div>
      ),
    },
    {
      title: "TechGear Electronics",
      content: (
        <div className="flex flex-col gap-8">
          <div className="w-full prose prose-lg dark:prose-invert">
            <p className="text-lg text-foreground/90 leading-relaxed">
              Implemented end-to-end AI automation for TechGear, a leading consumer electronics retailer with 500+ products.
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>50% reduction in customer support costs</li>
              <li>32% increase in average order value</li>
              <li>24/7 automated product recommendations</li>
              <li>95% accurate inventory forecasting</li>
            </ul>
            <blockquote className="border-l-4 border-brand pl-4 italic mt-4">
              "The AI system's ability to predict trends and automate customer support has given us a significant competitive advantage."
              <footer className="text-sm mt-2">- Michael Chang, CTO of TechGear</footer>
            </blockquote>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <ProjectImage src="https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=800&h=450&fit=crop" alt="TechGear store" />
            <ProjectImage src="https://images.unsplash.com/photo-1550009158-9ebf69173e03?w=800&h=450&fit=crop" alt="Electronics display" />
            <ProjectImage src="https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=800&h=450&fit=crop" alt="Tech support team" />
          </div>
        </div>
      ),
    },
    {
      title: "GlobalBank Financial",
      content: (
        <div className="flex flex-col gap-8">
          <div className="w-full prose prose-lg dark:prose-invert">
            <p className="text-lg text-foreground/90 leading-relaxed">
              Deployed AI automation across GlobalBank's retail banking operations, serving over 2 million customers.
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>75% reduction in query handling time</li>
              <li>42% increase in customer satisfaction</li>
              <li>90% accuracy in fraud detection</li>
              <li>28% reduction in operational costs</li>
            </ul>
            <blockquote className="border-l-4 border-brand pl-4 italic mt-4">
              "The AI automation has transformed our customer service while maintaining the highest security standards."
              <footer className="text-sm mt-2">- Emma Thompson, Head of Digital Banking</footer>
            </blockquote>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <ProjectImage src="https://images.unsplash.com/photo-1556742212-5b321f3c261b?w=800&h=450&fit=crop" alt="Bank interior" />
            <ProjectImage src="https://images.unsplash.com/photo-1601597111158-2fceff292cdc?w=800&h=450&fit=crop" alt="Digital banking" />
            <ProjectImage src="https://images.unsplash.com/photo-1563986768494-4dee2763ff3f?w=800&h=450&fit=crop" alt="Financial consulting" />
          </div>
        </div>
      ),
    },
    {
      title: "MVMT Watches",
      content: (
        <div className="flex flex-col gap-8">
          <div className="w-full prose prose-lg dark:prose-invert">
            <p className="text-lg text-foreground/90 leading-relaxed">
              Transformed MVMT's dropshipping watch business with AI-powered automation, scaling their operations globally while maintaining premium customer experience.
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>85% reduction in customer service response time</li>
              <li>42% increase in average order value through AI recommendations</li>
              <li>Automated order tracking across 180+ countries</li>
              <li>96% positive customer feedback rate</li>
            </ul>
            <blockquote className="border-l-4 border-brand pl-4 italic mt-4">
              "The AI automation platform has been a game-changer for our global operations. It's helped us scale efficiently while maintaining the premium experience our customers expect."
              <footer className="text-sm mt-2">- Jake Kassan, Co-founder of MVMT</footer>
            </blockquote>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <ProjectImage src="https://images.unsplash.com/photo-1523170335258-f5ed11844a49?w=800&h=450&fit=crop" alt="MVMT Watch collection" />
            <ProjectImage src="https://images.unsplash.com/photo-1542496658-e33a6d0d50f6?w=800&h=450&fit=crop" alt="Luxury watch display" />
            <ProjectImage src="https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=800&h=450&fit=crop" alt="Watch manufacturing" />
          </div>
        </div>
      ),
    },
  ];

  return (
    <div id="projects-section" className="w-full bg-background relative" ref={sectionRef}>
      {/* Background gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-background/95 to-background pointer-events-none" />
      
      <div className="container mx-auto relative z-10">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-brand/10 border border-brand/20 text-brand mb-4">
            <Trophy className="h-4 w-4" />
            <span className="text-sm font-medium">Success Stories</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500">
            Transforming Businesses with AI
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Discover how our AI automation solutions have revolutionized operations and driven growth.
          </p>
        </motion.div>

        {/* Timeline Section */}
        <div className="relative pt-16">
          <motion.div
            className="absolute left-[19px] top-0 bottom-0 w-[2px] bg-blue-500/20 rounded-full"
            style={{
              opacity: useTransform(scrollYProgress, [0, 0.5, 1], [0, 1, 0]),
              scale: useTransform(scrollYProgress, [0, 0.5, 1], [0.8, 1.2, 0.8]),
              filter: "blur(8px)",
            }}
          />
          <div className="relative z-10">
            <Timeline data={data} />
          </div>
        </div>

        {/* View Full Portfolio Button */}
        <div className="flex justify-center mt-16 mb-8">
          <a
            href="https://www.behance.net/gallery/219043635/AI-Automation-Chat-Voice-Email-Assistants"
            target="_blank"
            rel="noopener noreferrer"
          >
            <RainbowButton className="group relative flex items-center gap-2 px-6 py-3">
              <span className="relative z-10 flex items-center gap-2">
                View Full Portfolio on Behance
              </span>
            </RainbowButton>
          </a>
        </div>
      </div>
    </div>
  );
}

function ProjectImage({ src, alt }: { src: string; alt: string }) {
  return (
    <div className="relative aspect-video rounded-xl overflow-hidden border border-border/50">
      <Image src={src} alt={alt} width={800} height={450} className="object-cover w-full h-full" />
    </div>
  );
}