/**
 * Chat Message Component
 *
 * This component renders individual chat messages in the conversation.
 * It handles different message types (user vs. assistant) with appropriate styling,
 * and includes features for formatting message content.
 *
 * The component is designed to:
 * - Display messages with different styles based on sender
 * - Format message content with proper spacing and line breaks
 * - Show timestamps for each message
 * - Support future enhancements like syntax highlighting
 *
 * @module ChatMessage
 * @author Nada Mohamed
 */

import { cn } from "@/lib/utils";
import { MessageRole } from "@/types/message";
import { useEffect, useRef } from "react";

/**
 * Props for the ChatMessage component
 *
 * @interface ChatMessageProps
 * @property {MessageRole} role - The sender of the message ('user' or 'assistant')
 * @property {string} content - The text content of the message
 * @property {string} timestamp - The formatted time when the message was sent
 */
interface ChatMessageProps {
  role: MessageRole;
  content: string;
  timestamp: string;
}

/**
 * ChatMessage Component
 *
 * Renders a single message in the chat conversation with appropriate styling
 * based on whether it's from the user or the AI assistant.
 *
 * @param {ChatMessageProps} props - Component props
 * @returns {JSX.Element} The rendered chat message
 */
export function ChatMessage({ role, content, timestamp }: ChatMessageProps) {
  // Reference to the content div for potential DOM manipulation
  const contentRef = useRef<HTMLDivElement>(null);

  /**
   * Effect for handling formatted content
   *
   * This effect runs whenever the content changes and can be used to apply
   * special formatting to the message content, such as syntax highlighting
   * for code blocks or other rich text formatting.
   *
   * Currently, this is a placeholder for future enhancements, but the
   * infrastructure is in place to add these features when needed.
   */
  useEffect(() => {
    if (contentRef.current) {
      // Add syntax highlighting or other formatting if needed
      // This is a placeholder for future enhancements
      // Examples of potential enhancements:
      // - Code syntax highlighting
      // - Markdown rendering
      // - LaTeX formula rendering
      // - Link detection and formatting
    }
  }, [content]); // Re-run effect when content changes

  /**
   * Processes message content to improve formatting
   *
   * This function applies text transformations to improve readability:
   * - Replaces excessive consecutive newlines with just two newlines
   *   to prevent unnecessary large gaps in the message display
   *
   * Additional formatting could be added here in the future.
   *
   * @param {string} text - The raw message text
   * @returns {string} The processed message text with improved formatting
   */
  const processContent = (text: string) => {
    // Replace consecutive newlines (3 or more) with just 2 newlines
    // This prevents excessive vertical spacing while preserving paragraph breaks
    return text.replace(/\n{3,}/g, "\n\n");
  };

  /**
   * Render the chat message
   *
   * The message UI consists of:
   * 1. An outer container that positions the message (left or right)
   * 2. A message bubble with appropriate styling based on sender
   * 3. A header showing the sender name
   * 4. The message content with proper formatting
   * 5. A timestamp showing when the message was sent
   */
  return (
    <div
      className={cn(
        "flex w-full mb-4",
        // Align user messages to the right, assistant messages to the left
        role === "user" ? "justify-end" : "justify-start",
        // Add animation for message appearance
        "animate-message-fade-in motion-reduce:animate-none"
      )}
    >
      {/* Message bubble container */}
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3",
          // Different styling for user vs. assistant messages
          role === "user"
            ? "bg-primary text-primary-foreground" // User messages use primary color
            : "glass", // Assistant messages use glass effect
          // Add slide animation based on message sender
          role === "user"
            ? "animate-slide-from-right motion-reduce:animate-none"
            : "animate-slide-from-left motion-reduce:animate-none",
          // Add hover effect for better interactivity
          "hover:shadow-md transition-shadow duration-300"
        )}
        style={{
          // Hardware acceleration for smoother animations
          transform: "translateZ(0)",
          backfaceVisibility: "hidden",
        }}
      >
        {/* Message header with sender name */}
        <div className="mb-1">
          <div className="flex items-center">
            <span className="font-semibold chat-message-sender">
              {role === "user" ? "You" : "Jarvis"}
            </span>
          </div>
        </div>

        {/* Message content container with prose styling for rich text */}
        <div
          ref={contentRef} // Reference for potential DOM manipulation
          className="prose dark:prose-invert max-w-none overflow-auto scrollbar-themed"
        >
          {/* Actual message text with formatting */}
          <div
            className="m-0 whitespace-pre-wrap break-words chat-message-content"
            style={{ lineHeight: "var(--line-height-normal)" }}
          >
            {/* Process the content to improve formatting */}
            {processContent(content)}
          </div>
        </div>

        {/* Message timestamp */}
        <div className="mt-1 opacity-70 chat-message-timestamp">
          {timestamp}
        </div>
      </div>
    </div>
  );
}
