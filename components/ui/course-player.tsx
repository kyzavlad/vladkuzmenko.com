"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Clock, Play } from "lucide-react";
import VideoPlayer from "@/components/ui/video-player";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface Lesson {
  id: string;
  title: string;
  duration: string;
  videoUrl: string;
}

interface Course {
  id: string;
  title: string;
  description: string;
  lessons: Lesson[];
}

export function CoursePlayer({ course }: { course: Course }) {
  const [selectedLesson, setSelectedLesson] = useState(course.lessons[0]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <div className="lg:col-span-2">
        <VideoPlayer src={selectedLesson.videoUrl} />
        <div className="mt-6">
          <h2 className="text-2xl font-bold text-white mb-2">{course.title}</h2>
          <p className="text-gray-400">{course.description}</p>
        </div>
      </div>

      <div className="space-y-4">
        <Card className="bg-[#2f3136] border-neutral-700">
          <div className="p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Course Lessons</h3>
            <div className="space-y-2">
              {course.lessons.map((lesson, index) => (
                <motion.div
                  key={lesson.id}
                  onClick={() => setSelectedLesson(lesson)}
                  className={cn(
                    "w-full p-3 rounded-lg flex items-center gap-3 transition-all duration-300 cursor-pointer",
                    selectedLesson.id === lesson.id
                      ? "bg-brand/20 text-brand"
                      : "hover:bg-neutral-700/50 text-white"
                  )}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex-shrink-0">
                    <div
                      className={cn(
                        "h-8 w-8 rounded-lg flex items-center justify-center",
                        selectedLesson.id === lesson.id && "text-brand"
                      )}
                    >
                      <Play className="h-4 w-4" />
                    </div>
                  </div>
                  <div className="flex-1 text-left">
                    <div className="font-medium">
                      Lesson {index + 1}: {lesson.title}
                    </div>
                    <div className="text-sm text-gray-400 flex items-center gap-1 mt-1">
                      <Clock className="h-3 w-3" />
                      {lesson.duration}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Card>

        <Card className="bg-[#2f3136] border-neutral-700 p-4">
          <h3 className="text-lg font-semibold text-white mb-4">Course Progress</h3>
          <Progress value={33} className="mb-2" />
          <p className="text-sm text-gray-400">2 of 6 lessons completed</p>
        </Card>
      </div>
    </div>
  );
}