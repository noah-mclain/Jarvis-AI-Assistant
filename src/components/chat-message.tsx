import { cn } from "@/lib/utils";
import { MessageRole } from "@/types/message";
import { useEffect, useRef } from "react";

interface ChatMessageProps {
  role: MessageRole;
  content: string;
  timestamp: string;
}

export function ChatMessage({ role, content, timestamp }: ChatMessageProps) {
  const contentRef = useRef<HTMLDivElement>(null);

  // Ensure code blocks and other formatted content display properly
  useEffect(() => {
    if (contentRef.current) {
      // Add syntax highlighting or other formatting if needed
      // This is a placeholder for future enhancements
    }
  }, [content]);

  // Process content to handle special formatting
  const processContent = (text: string) => {
    // Replace consecutive newlines with proper spacing
    return text.replace(/\n{3,}/g, "\n\n");
  };

  return (
    <div
      className={cn(
        "flex w-full mb-4",
        role === "user" ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3",
          role === "user" ? "bg-primary text-primary-foreground" : "glass"
        )}
      >
        <div className="mb-1">
          <div className="flex items-center">
            <span className="font-semibold chat-message-sender">
              {role === "user" ? "You" : "Jarvis"}
            </span>
          </div>
        </div>
        <div
          ref={contentRef}
          className="prose dark:prose-invert max-w-none overflow-auto scrollbar-themed"
        >
          <div
            className="m-0 whitespace-pre-wrap break-words chat-message-content"
            style={{ lineHeight: "var(--line-height-normal)" }}
          >
            {processContent(content)}
          </div>
        </div>
        <div className="mt-1 opacity-70 chat-message-timestamp">
          {timestamp}
        </div>
      </div>
    </div>
  );
}
