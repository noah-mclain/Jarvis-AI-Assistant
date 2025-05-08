import { createContext, useContext, useEffect, useState } from "react";
import { useWindowSize } from "@/hooks/use-window-size";

// Define min and max font sizes
const MIN_FONT_SIZE = 12;
const MAX_FONT_SIZE = 24;
const DEFAULT_FONT_SIZE = 18; // Default base font size (--font-size-base)

// Calculate dynamic font size based on window width
const calculateDynamicFontSize = (width: number | undefined): number => {
  if (!width) return DEFAULT_FONT_SIZE;

  // Base calculation on window width
  // For smaller screens (mobile), use smaller font
  // For larger screens, use larger font
  const minWidth = 320; // Minimum width to consider
  const maxWidth = 1920; // Maximum width to consider

  // Clamp width between min and max
  const clampedWidth = Math.max(minWidth, Math.min(width, maxWidth));

  // Linear interpolation between MIN_FONT_SIZE and MAX_FONT_SIZE
  const ratio = (clampedWidth - minWidth) / (maxWidth - minWidth);
  const dynamicSize = MIN_FONT_SIZE + ratio * (MAX_FONT_SIZE - MIN_FONT_SIZE);

  // Round to nearest 0.5
  return Math.round(dynamicSize * 2) / 2;
};

type FontSizeProviderProps = {
  children: React.ReactNode;
  storageKey?: string;
};

type FontSizeProviderState = {
  fontSize: number;
  increaseFontSize: () => void;
  decreaseFontSize: () => void;
  resetFontSize: () => void;
};

const initialState: FontSizeProviderState = {
  fontSize: DEFAULT_FONT_SIZE,
  increaseFontSize: () => null,
  decreaseFontSize: () => null,
  resetFontSize: () => null,
};

const FontSizeProviderContext =
  createContext<FontSizeProviderState>(initialState);

export function FontSizeProvider({
  children,
  storageKey = "jarvis-ui-font-size",
  ...props
}: FontSizeProviderProps) {
  const windowSize = useWindowSize();
  const [dynamicFontSize, setDynamicFontSize] =
    useState<number>(DEFAULT_FONT_SIZE);
  const [userFontSizeAdjustment, setUserFontSizeAdjustment] =
    useState<number>(0);

  // Calculate the base dynamic font size whenever window size changes
  useEffect(() => {
    const calculatedSize = calculateDynamicFontSize(windowSize.width);
    setDynamicFontSize(calculatedSize);
  }, [windowSize.width]);

  // Load user's font size adjustment from localStorage
  useEffect(() => {
    const savedAdjustment = localStorage.getItem(storageKey);
    if (savedAdjustment !== null) {
      setUserFontSizeAdjustment(parseFloat(savedAdjustment));
    }
  }, [storageKey]);

  // Calculate the final font size (dynamic base + user adjustment)
  const finalFontSize = Math.max(
    MIN_FONT_SIZE,
    Math.min(MAX_FONT_SIZE, dynamicFontSize + userFontSizeAdjustment)
  );

  // Apply the font size to CSS variables
  useEffect(() => {
    const root = document.documentElement;

    // Calculate relative sizes based on the base font size
    const baseSize = finalFontSize;
    const xsSize = baseSize * 0.78; // 0.875 relative to base
    const smSize = baseSize * 0.89; // 1.0 relative to base
    const lgSize = baseSize * 1.11; // 1.25 relative to base
    const xlSize = baseSize * 1.33; // 1.5 relative to base
    const xxlSize = baseSize * 1.56; // 1.75 relative to base
    const xxxlSize = baseSize * 1.78; // 2.0 relative to base

    // Update CSS variables
    root.style.setProperty("--font-size-xs", `${xsSize}px`);
    root.style.setProperty("--font-size-sm", `${smSize}px`);
    root.style.setProperty("--font-size-base", `${baseSize}px`);
    root.style.setProperty("--font-size-lg", `${lgSize}px`);
    root.style.setProperty("--font-size-xl", `${xlSize}px`);
    root.style.setProperty("--font-size-2xl", `${xxlSize}px`);
    root.style.setProperty("--font-size-3xl", `${xxxlSize}px`);

    // Apply font size to the body element to affect all text
    root.style.setProperty("font-size", `${baseSize}px`);

    // Update line heights to maintain aspect ratios
    const tightLineHeight = 1.2;
    const normalLineHeight = 1.5;
    const relaxedLineHeight = 1.75;

    root.style.setProperty("--line-height-tight", `${tightLineHeight}`);
    root.style.setProperty("--line-height-normal", `${normalLineHeight}`);
    root.style.setProperty("--line-height-relaxed", `${relaxedLineHeight}`);

    // Apply a CSS class to the body to trigger font size updates
    document.body.classList.add("font-size-updated");
  }, [finalFontSize]);

  // Save user adjustment to localStorage when it changes
  useEffect(() => {
    localStorage.setItem(storageKey, userFontSizeAdjustment.toString());
  }, [userFontSizeAdjustment, storageKey]);

  // Increase font size by 1px
  const increaseFontSize = () => {
    if (finalFontSize < MAX_FONT_SIZE) {
      setUserFontSizeAdjustment((prev) => prev + 1);
    }
  };

  // Decrease font size by 1px
  const decreaseFontSize = () => {
    if (finalFontSize > MIN_FONT_SIZE) {
      setUserFontSizeAdjustment((prev) => prev - 1);
    }
  };

  // Reset to dynamic font size (remove user adjustment)
  const resetFontSize = () => {
    setUserFontSizeAdjustment(0);
  };

  // Set up keyboard shortcuts for font size adjustment
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Check for Ctrl/Cmd + Plus or Ctrl/Cmd + Equals (same key)
      if (
        (event.ctrlKey || event.metaKey) &&
        (event.key === "+" || event.key === "=")
      ) {
        event.preventDefault();
        increaseFontSize();
      }
      // Check for Ctrl/Cmd + Minus
      else if ((event.ctrlKey || event.metaKey) && event.key === "-") {
        event.preventDefault();
        decreaseFontSize();
      }
      // Check for Ctrl/Cmd + 0 (reset)
      else if ((event.ctrlKey || event.metaKey) && event.key === "0") {
        event.preventDefault();
        resetFontSize();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [finalFontSize]);

  const value = {
    fontSize: finalFontSize,
    increaseFontSize,
    decreaseFontSize,
    resetFontSize,
  };

  return (
    <FontSizeProviderContext.Provider {...props} value={value}>
      {children}
    </FontSizeProviderContext.Provider>
  );
}

export const useFontSize = () => {
  const context = useContext(FontSizeProviderContext);

  if (context === undefined)
    throw new Error("useFontSize must be used within a FontSizeProvider");

  return context;
};
