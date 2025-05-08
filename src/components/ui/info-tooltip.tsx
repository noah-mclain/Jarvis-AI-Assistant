import React from "react";
import { HelpCircle } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface InfoTooltipProps {
  content: React.ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
  className?: string;
  iconClassName?: string;
  iconSize?: number;
}

export function InfoTooltip({
  content,
  side = "top",
  align = "center",
  className,
  iconClassName,
  iconSize = 16,
}: InfoTooltipProps) {
  return (
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>
        <button
          type="button"
          className={cn(
            "inline-flex items-center justify-center rounded-full text-muted-foreground hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            iconClassName
          )}
          aria-label="More information"
        >
          <HelpCircle size={iconSize} />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side={side}
        align={align}
        className={cn(
          "max-w-xs p-4 glass rounded-xl shadow-lg border border-border",
          "text-sm text-foreground",
          "animate-in zoom-in-90 duration-300",
          className
        )}
      >
        {content}
      </TooltipContent>
    </Tooltip>
  );
}
