"use client";

import { useState, useRef, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Send, Lock } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  id: string;
  user: {
    name: string;
    avatar: string;
  };
  content: string;
  timestamp: string;
}

interface ChatWindowProps {
  channelName: string;
  messages: Message[];
  onSendMessage: (content: string) => void;
}

export function ChatWindow({ channelName, messages: initialMessages, onSendMessage }: ChatWindowProps) {
  const [messages, setMessages] = useState(initialMessages);
  const [newMessage, setNewMessage] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (newMessage.trim()) {
      const message = {
        id: `msg-${Date.now()}`,
        user: {
          name: "You",
          avatar: "https://github.com/shadcn.png"
        },
        content: newMessage,
        timestamp: "Just now"
      };
      setMessages([...messages, message]);
      onSendMessage(newMessage);
      setNewMessage("");
    }
  };

  return (
    <Card className="bg-[#2f3136] border-neutral-700">
      <div className="p-4 border-b border-neutral-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Lock className="h-4 w-4 text-green-500" />
          <h2 className="font-semibold text-white">{channelName}</h2>
        </div>
      </div>

      <div className="h-[600px] flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex items-start gap-3"
              >
                <Avatar>
                  <AvatarImage src={message.user.avatar} />
                  <AvatarFallback>{message.user.name[0]}</AvatarFallback>
                </Avatar>
                <div>
                  <div className="flex items-baseline gap-2">
                    <span className="font-semibold text-white">{message.user.name}</span>
                    <span className="text-xs text-gray-400">{message.timestamp}</span>
                  </div>
                  <p className="text-sm mt-1 text-gray-300">{message.content}</p>
                </div>
              </motion.div>
            ))}
            <div ref={messagesEndRef} />
          </AnimatePresence>
        </div>

        <div className="p-4 border-t border-neutral-700">
          <div className="flex items-center gap-2">
            <Input
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Type a message..."
              className="bg-neutral-800 border-neutral-700 text-white"
            />
            <button
              onClick={handleSend}
              className="p-2 rounded-full bg-brand text-white hover:bg-brand/90 transition-colors"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </Card>
  );
}