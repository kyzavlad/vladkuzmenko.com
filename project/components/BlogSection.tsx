"use client";

import { blogArticles } from "@/app/blog/data";
import DisplayCards from "@/components/ui/display-cards";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog";

export function BlogSection() {
  const cards = blogArticles.map((article) => ({
    icon: article.icon,
    title: article.title,
    description: article.description,
    date: article.date,
    iconClassName: article.iconClassName,
    titleClassName: article.titleClassName,
    wrapper: (cardInner: React.ReactNode) => (
      <Dialog key={article.id}>
        <DialogTrigger asChild>
          {cardInner}
        </DialogTrigger>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-3 text-2xl">
              {article.icon}
              {article.title}
            </DialogTitle>
          </DialogHeader>
          <div className="mt-4">
            <div className="flex items-center gap-4 mb-4 text-sm text-muted-foreground">
              <span>{article.date}</span>
              <span>•</span>
              <span>{article.category}</span>
              <span>•</span>
              <span>{article.readTime}</span>
            </div>
            <img 
              src={article.image} 
              alt={article.title}
              className="w-full h-[300px] object-cover rounded-lg mb-6"
            />
            <div className="prose prose-lg dark:prose-invert max-w-none">
              <div dangerouslySetInnerHTML={{ __html: article.content }} />
            </div>
          </div>
        </DialogContent>
      </Dialog>
    )
  }));

  return (
    <div id="blog-section" className="w-full py-16 md:py-32 bg-background">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gradient">
            AI Automation Insights
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Explore our latest articles, case studies, and best practices 
            for implementing AI automation in your business.
          </p>
        </div>

        <div className="flex flex-col items-center justify-center">
          <div className="w-full max-w-[1200px] mx-auto mb-20">
            <DisplayCards cards={cards} />
          </div>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div className="bg-background/50 p-6 rounded-lg border border-border/50 shadow-sm">
            <h3 className="text-xl font-semibold mb-3">Latest Research</h3>
            <p className="text-muted-foreground">
              Stay updated with our cutting-edge research on AI automation 
              technologies and their practical applications.
            </p>
          </div>
          <div className="bg-background/50 p-6 rounded-lg border border-border/50 shadow-sm">
            <h3 className="text-xl font-semibold mb-3">Success Stories</h3>
            <p className="text-muted-foreground">
              Read how businesses across various industries have successfully 
              implemented our AI automation solutions.
            </p>
          </div>
          <div className="bg-background/50 p-6 rounded-lg border border-border/50 shadow-sm">
            <h3 className="text-xl font-semibold mb-3">Implementation Guides</h3>
            <p className="text-muted-foreground">
              Step-by-step guides to help you seamlessly integrate AI automation 
              into your existing business processes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}