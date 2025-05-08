/**
 * Chat Input Component
 *
 * This component provides a textarea input for users to type messages to the AI assistant.
 * It includes features like:
 * - Auto-resizing textarea that grows with content
 * - Submit button that enables/disables based on input state
 * - Keyboard shortcuts (Enter to send, Shift+Enter for new line)
 * - Loading and disabled states
 * - Custom placeholder text
 *
 * The component is designed to work offline and integrates with the main chat interface.
 *
 * @module ChatInput
 * @author Nada Mohamed
 */

import { useState, useRef, FormEvent, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { TextareaAutosize } from "@/components/ui/textarea-autosize";

/**
 * Props for the ChatInput component
 *
 * @interface ChatInputProps
 * @property {function} onSend - Callback function that receives the message text when sent
 * @property {boolean} [isLoading=false] - Whether the application is currently processing a message
 * @property {boolean} [disabled=false] - Whether the input should be disabled
 * @property {string} [placeholder="Ask Jarvis anything..."] - Placeholder text for the textarea
 */
interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

/**
 * ChatInput Component
 *
 * A textarea input component that allows users to send messages to the AI assistant.
 * The component handles form submission, keyboard shortcuts, and manages input state.
 *
 * @param {ChatInputProps} props - Component props
 * @returns {JSX.Element} The rendered chat input component
 */
export function ChatInput({
  onSend,
  isLoading = false,
  disabled = false,
  placeholder = "Ask Jarvis anything...",
}: ChatInputProps) {
  // State to track the current input value
  const [input, setInput] = useState("");

  // Reference to the textarea element for direct DOM manipulation
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  /**
   * Effect to reset textarea height when input is cleared
   *
   * This ensures the textarea returns to its default height when the user
   * sends a message or clears the input, providing a consistent UI experience.
   */
  useEffect(() => {
    if (input === "" && textareaRef.current) {
      // Reset height to default when input is cleared
      textareaRef.current.style.height = "70px";
    }
  }, [input]); // Re-run effect when input changes

  /**
   * Handles form submission
   *
   * This function:
   * 1. Prevents the default form submission behavior
   * 2. Validates that the input is not empty and the component is not disabled
   * 3. Calls the onSend callback with the input value
   * 4. Clears the input field
   *
   * @param {FormEvent} e - The form submission event
   */
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading && !disabled) {
      onSend(input);
      setInput("");
    }
  };

  /**
   * Handles keyboard events in the textarea
   *
   * This function enables the Enter key to submit the form, while
   * Shift+Enter creates a new line. This matches the behavior of
   * many modern chat applications.
   *
   * @param {React.KeyboardEvent<HTMLTextAreaElement>} e - The keyboard event
   */
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  /**
   * Render the chat input component
   *
   * The component consists of:
   * 1. A form wrapper to handle submission
   * 2. A glass-effect container for visual styling
   * 3. An auto-resizing textarea for message input
   * 4. A submit button that changes state based on input/loading
   */
  return (
    <form onSubmit={handleSubmit} className="relative">
      {/* Glass-effect container with rounded corners */}
      <div
        className={`glass rounded-2xl p-2 ${
          disabled ? "opacity-70" : ""
        } transition-all duration-300`}
        style={{
          // Hardware acceleration for smoother animations
          transform: "translateZ(0)",
          backfaceVisibility: "hidden",
          boxShadow: input.trim() ? "0 4px 12px rgba(0, 0, 0, 0.05)" : "none",
        }}
      >
        {/*
          Auto-resizing textarea component
          This textarea grows as the user types more content, up to a maximum number of rows
        */}
        <TextareaAutosize
          ref={textareaRef} // Reference for DOM manipulation
          value={input} // Controlled input value
          onChange={(e) => setInput(e.target.value)} // Update state on change
          onKeyDown={handleKeyDown} // Handle keyboard shortcuts
          placeholder={placeholder} // Custom placeholder text
          className="min-h-[70px] border-none bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/70 p-3 scrollbar-themed chat-input transition-all duration-300"
          maxRows={8} // Maximum number of visible rows
          disabled={disabled} // Disable when needed
          style={{
            // Show scrollbar only when content exceeds 8 rows
            overflowY: input.split("\n").length > 8 ? "auto" : "hidden",
            overflowX: "hidden", // Never show horizontal scrollbar
            lineHeight: "var(--line-height-normal)", // Use theme-defined line height
            // Hardware acceleration for smoother animations
            transform: "translateZ(0)",
            backfaceVisibility: "hidden",
          }}
        />

        {/* Button container aligned to the right */}
        <div className="flex justify-end p-1">
          {/*
            Submit button
            Disabled when:
            - The input is empty
            - A message is currently being processed
            - The component is explicitly disabled
          */}
          <Button
            type="submit"
            size="sm"
            disabled={isLoading || !input.trim() || disabled}
            className={`rounded-full transition-all duration-300 transform ${
              input.trim() && !isLoading && !disabled
                ? "hover:scale-105 active:scale-95"
                : ""
            }`}
            style={{
              // Hardware acceleration for smoother animations
              transform: "translateZ(0)",
              backfaceVisibility: "hidden",
            }}
          >
            {isLoading ? (
              <span className="flex items-center">
                <span className="mr-1">Processing</span>
                <span className="inline-flex">
                  <span className="animate-[pulse_0.8s_ease-in-out_infinite]">
                    .
                  </span>
                  <span className="animate-[pulse_0.8s_ease-in-out_0.2s_infinite]">
                    .
                  </span>
                  <span className="animate-[pulse_0.8s_ease-in-out_0.4s_infinite]">
                    .
                  </span>
                </span>
              </span>
            ) : (
              "Send"
            )}
          </Button>
        </div>
      </div>
    </form>
  );
}
