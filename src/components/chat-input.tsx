import { useState, useRef, FormEvent, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { TextareaAutosize } from "@/components/ui/textarea-autosize";

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  isLoading = false,
  disabled = false,
  placeholder = "Ask Jarvis anything...",
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Reset textarea height when input is cleared
  useEffect(() => {
    if (input === "" && textareaRef.current) {
      // Reset height to default when input is cleared
      textareaRef.current.style.height = "70px";
    }
  }, [input]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading && !disabled) {
      onSend(input);
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className={`glass rounded-2xl p-2 ${disabled ? "opacity-70" : ""}`}>
        <TextareaAutosize
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="min-h-[70px] border-none bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/70 p-3 scrollbar-themed chat-input"
          maxRows={8}
          disabled={disabled}
          style={{
            overflowY: input.split("\n").length > 8 ? "auto" : "hidden",
            overflowX: "hidden",
            lineHeight: "var(--line-height-normal)",
          }}
        />
        <div className="flex justify-end p-1">
          <Button
            type="submit"
            size="sm"
            disabled={isLoading || !input.trim() || disabled}
            className="rounded-full"
          >
            {isLoading ? "Processing..." : "Send"}
          </Button>
        </div>
      </div>
    </form>
  );
}
