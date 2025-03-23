"use client";

import { Header } from "@/components/ui/header";
import { FooterSection } from "@/components/FooterSection";
import { BentoGrid } from "@/components/ui/bento-grid";
import { TrendingUp, CheckCircle2, Video, Globe } from "lucide-react";

export default function DashboardPage() {
  const dashboardItems = [
    {
      title: "Learning Progress",
      meta: "85% Complete",
      description: "Track your course completion and achievements across all academies",
      icon: <TrendingUp className="w-4 h-4 text-blue-500" />,
      status: "Active",
      tags: ["Courses", "Progress", "Goals"],
      colSpan: 2,
      hasPersistentHover: true,
    },
    {
      title: "Completed Tasks",
      meta: "24 this week",
      description: "View your completed assignments and daily tasks",
      icon: <CheckCircle2 className="w-4 h-4 text-emerald-500" />,
      status: "Updated",
      tags: ["Tasks", "Achievements"],
    },
    {
      title: "Course Library",
      meta: "150+ courses",
      description: "Access all available courses and learning materials",
      icon: <Video className="w-4 h-4 text-purple-500" />,
      tags: ["Library", "Content"],
      colSpan: 2,
    },
    {
      title: "Community Stats",
      meta: "5.2k members",
      description: "Connect with fellow learners and track community engagement",
      icon: <Globe className="w-4 h-4 text-sky-500" />,
      status: "Live",
      tags: ["Community", "Network"],
    },
  ];

  return (
    <main className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto py-12">
        <h1 className="text-3xl font-bold mb-8">Dashboard</h1>
        <BentoGrid items={dashboardItems} />
      </div>
      <FooterSection />
    </main>
  );
}