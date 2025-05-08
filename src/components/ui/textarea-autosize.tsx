import React, { useState, useRef, useEffect } from "react";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

export interface TextareaAutosizeProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  maxRows?: number;
}

const TextareaAutosize = React.forwardRef<
  HTMLTextAreaElement,
  TextareaAutosizeProps
>(({ className, maxRows = 5, onChange, ...props }, ref) => {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [lineHeight, setLineHeight] = useState<number>(0);

  // Update line height calculation on mount
  useEffect(() => {
    if (textareaRef.current) {
      const style = window.getComputedStyle(textareaRef.current);
      const calculatedLineHeight = parseInt(style.lineHeight);
      // Use a larger default line height (28px) to accommodate larger font sizes
      setLineHeight(isNaN(calculatedLineHeight) ? 28 : calculatedLineHeight);
    }
  }, []);

  // Handle auto-resizing
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = e.target;

    if (lineHeight && textarea) {
      // Store the current scroll position
      const scrollTop =
        window.pageYOffset || document.documentElement.scrollTop;

      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = "auto";

      // Calculate rows
      const currentRows = textarea.scrollHeight / lineHeight;

      if (currentRows <= maxRows) {
        textarea.style.height = `${textarea.scrollHeight}px`;
      } else {
        textarea.style.height = `${lineHeight * maxRows}px`;
      }

      // Restore the scroll position to prevent page jumping
      window.scrollTo(0, scrollTop);
    }

    if (onChange) {
      onChange(e);
    }
  };

  // Auto-resize on initial content and window resize
  useEffect(() => {
    if (textareaRef.current && lineHeight) {
      const textarea = textareaRef.current;
      textarea.style.height = "auto";

      const currentRows = textarea.scrollHeight / lineHeight;
      if (currentRows <= maxRows) {
        textarea.style.height = `${textarea.scrollHeight}px`;
      } else {
        textarea.style.height = `${lineHeight * maxRows}px`;
      }
    }

    const handleResize = () => {
      if (textareaRef.current && lineHeight) {
        const textarea = textareaRef.current;
        textarea.style.height = "auto";

        const currentRows = textarea.scrollHeight / lineHeight;
        if (currentRows <= maxRows) {
          textarea.style.height = `${textarea.scrollHeight}px`;
        } else {
          textarea.style.height = `${lineHeight * maxRows}px`;
        }
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [lineHeight, maxRows, props.value]);

  return (
    <Textarea
      ref={(element) => {
        // Handle both our ref and the forwarded ref
        textareaRef.current = element;
        if (typeof ref === "function") {
          ref(element);
        } else if (ref) {
          ref.current = element;
        }
      }}
      className={cn("resize-none overflow-y-auto scrollbar-themed", className)}
      onChange={handleChange}
      style={{
        scrollbarWidth: "thin",
        scrollbarColor: "var(--primary) transparent",
        ...props.style,
      }}
      {...props}
    />
  );
});

TextareaAutosize.displayName = "TextareaAutosize";

export { TextareaAutosize };
