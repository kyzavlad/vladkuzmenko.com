"use client";

import { useState, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Pause, Volume2, VolumeX } from "lucide-react";
import { cn } from "@/lib/utils";

// Voice sample data with correct audio file paths
const voiceSamples = [
  {
    id: "assistant-professional",
    name: "Professional Assistant",
    description: "Clear, professional tone with natural inflection",
    audioSrc: "/0304(1).MP3",
    color: "brand",
    features: ["Natural language understanding", "Clear pronunciation", "Professional tone", "Perfect for business"]
  },
  {
    id: "assistant-friendly",
    name: "Friendly Guide",
    description: "Warm, approachable voice with conversational style",
    audioSrc: "/0304(2).MP3",
    color: "color-1",
    features: ["Conversational tone", "Friendly approach", "Natural pauses", "Engaging style"]
  },
  {
    id: "assistant-technical",
    name: "Technical Expert",
    description: "Precise, authoritative voice for technical explanations",
    audioSrc: "/0305.MP3",
    color: "color-3",
    features: ["Technical accuracy", "Clear articulation", "Authoritative tone", "Complex terminology"]
  }
];

export function AudioSection() {
  const [currentlyPlaying, setCurrentlyPlaying] = useState<string | null>(null);
  const [isMuted, setIsMuted] = useState(false);
  const audioRefs = useRef<{ [key: string]: HTMLAudioElement | null }>({});

  // Play a specific voice sample
  const playSample = (sampleId: string) => {
    // Stop any currently playing audio
    if (currentlyPlaying && audioRefs.current[currentlyPlaying]) {
      audioRefs.current[currentlyPlaying]?.pause();
      if (audioRefs.current[currentlyPlaying]) {
        audioRefs.current[currentlyPlaying]!.currentTime = 0;
      }
    }
    
    // If we're clicking on the already playing sample, just stop it
    if (currentlyPlaying === sampleId) {
      setCurrentlyPlaying(null);
      return;
    }
    
    // Play the new sample
    const sample = voiceSamples.find(s => s.id === sampleId);
    if (sample) {
      let audio = audioRefs.current[sampleId];
      
      if (!audio) {
        audio = new Audio(sample.audioSrc);
        audio.preload = "auto";
        audioRefs.current[sampleId] = audio;
      }

      audio.currentTime = 0;
      audio.muted = isMuted;
      
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.then(() => {
          setCurrentlyPlaying(sampleId);
        }).catch(() => {
          // Silently handle any play errors
          setCurrentlyPlaying(null);
        });
      }
    }
  };

  // Toggle mute for all audio
  const toggleMute = () => {
    setIsMuted(!isMuted);
    Object.values(audioRefs.current).forEach(audio => {
      if (audio) {
        audio.muted = !isMuted;
      }
    });
  };

  // Handle audio ending
  const handleAudioEnded = (sampleId: string) => {
    if (currentlyPlaying === sampleId) {
      setCurrentlyPlaying(null);
    }
  };

  return (
    <div id="audio-section" className="w-full py-16 md:py-24 bg-background">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gradient">Voice-Enabled AI Assistant</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Experience our AI voices that power our automation solutions. Listen to different voice styles and find the perfect match for your business needs.
          </p>
        </div>
        
        <div className="max-w-3xl mx-auto">
          <Card className="border-brand/10 dark:border-brand/5 shadow-lg">
            <CardHeader className="text-center">
              <CardTitle>AI Voice Assistant</CardTitle>
              <CardDescription>Click to preview our AI assistant voices</CardDescription>
            </CardHeader>
            <CardContent>
              {/* Voice samples list */}
              <div className="space-y-4">
                {voiceSamples.map((sample) => (
                  <div 
                    key={sample.id}
                    className={cn(
                      "flex flex-col gap-4 p-4 rounded-lg transition-all duration-300",
                      "border hover:border-opacity-50",
                      currentlyPlaying === sample.id 
                        ? `border-${sample.color} bg-${sample.color}/5` 
                        : "border-border/50 hover:bg-muted/30"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className={cn(
                            "h-10 w-10 p-0 rounded-full",
                            `text-${sample.color} hover:text-${sample.color} hover:bg-${sample.color}/10`
                          )}
                          onClick={() => playSample(sample.id)}
                        >
                          {currentlyPlaying === sample.id ? (
                            <Pause className="h-4 w-4" />
                          ) : (
                            <Play className="h-4 w-4 ml-0.5" />
                          )}
                        </Button>
                        <div className="text-left">
                          <p className="font-medium">{sample.name}</p>
                          <p className="text-sm text-muted-foreground">{sample.description}</p>
                        </div>
                      </div>
                      
                      {/* Visualizer for currently playing sample */}
                      {currentlyPlaying === sample.id && (
                        <div className="flex space-x-1 items-center">
                          {[...Array(4)].map((_, i) => (
                            <div
                              key={i}
                              className={`w-1 bg-${sample.color} rounded-full animate-pulse`}
                              style={{
                                height: `${12 + Math.random() * 12}px`,
                                animationDuration: `${0.7 + Math.random() * 0.5}s`,
                                animationDelay: `${i * 0.1}s`,
                              }}
                            />
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Features grid */}
                    <div className="grid grid-cols-2 gap-2">
                      {sample.features.map((feature, idx) => (
                        <div 
                          key={idx} 
                          className={cn(
                            "flex items-center gap-2 p-2 rounded-lg",
                            `bg-${sample.color}/5 text-sm`
                          )}
                        >
                          <div className={`w-1.5 h-1.5 rounded-full bg-${sample.color}`} />
                          {feature}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Mute button */}
              <div className="flex justify-end mt-6">
                <Button 
                  variant="ghost"
                  size="sm"
                  onClick={toggleMute}
                  className="text-muted-foreground hover:text-foreground"
                >
                  {isMuted ? (
                    <VolumeX className="h-4 w-4" />
                  ) : (
                    <Volume2 className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
          
          <div className="mt-8 text-center">
            <p className="text-sm text-muted-foreground">
              Our AI voices are designed to provide natural, human-like interactions across multiple languages and accents.
              <br />Custom voice development is available for enterprise clients.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}