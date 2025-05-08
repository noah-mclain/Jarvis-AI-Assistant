import { createContext, useContext, useEffect, useState } from "react";
import { hexToHsl } from "@/lib/color-utils";

type Theme = "dark" | "light" | "system";

type ThemeProviderProps = {
  children: React.ReactNode;
  defaultTheme?: Theme;
  storageKey?: string;
};

type ThemeProviderState = {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  accentColor: string;
  setAccentColor: (color: string) => void;
  primaryColor: string;
  setPrimaryColor: (color: string) => void;
  backgroundColor: string;
  setBackgroundColor: (color: string) => void;
  textColor: string;
  setTextColor: (color: string) => void;
  resetColorsToDefaults: () => void;
};

const initialState: ThemeProviderState = {
  theme: "system",
  setTheme: () => null,
  accentColor: "#a7c6f9", // Pastel blue for light mode
  setAccentColor: () => null,
  primaryColor: "#6fa0e3", // Pastel blue for light mode
  setPrimaryColor: () => null,
  backgroundColor: "#ffffff", // Pure white for light mode
  setBackgroundColor: () => null,
  textColor: "#4a3f35", // Soothing dark brown for light mode
  setTextColor: () => null,
  resetColorsToDefaults: () => null,
};

const ThemeProviderContext = createContext<ThemeProviderState>(initialState);

export function ThemeProvider({
  children,
  defaultTheme = "system",
  storageKey = "jarvis-ui-theme",
  ...props
}: ThemeProviderProps) {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem(storageKey) as Theme) || defaultTheme
  );

  // Default colors for light and dark modes
  const defaultColors = {
    light: {
      primary: "#6fa0e3", // Pastel blue
      accent: "#a7c6f9", // Lighter pastel blue
      background: "#ffffff", // Pure white
      text: "#4a3f35", // Soothing dark brown
    },
    dark: {
      primary: "#3b5f8a", // Neutral blue
      accent: "#4a6c9b", // Slightly lighter neutral blue
      background: "#121212", // Near black
      text: "#d0d0d0", // Soothing light gray
    },
  };

  // Get current mode (light/dark) based on theme setting
  const getCurrentMode = (): "light" | "dark" => {
    if (theme === "system") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    }
    return theme as "light" | "dark";
  };

  const [accentColor, setAccentColor] = useState<string>(
    () =>
      localStorage.getItem("jarvis-ui-accent") ||
      defaultColors[getCurrentMode()].accent
  );

  const [primaryColor, setPrimaryColor] = useState<string>(
    () =>
      localStorage.getItem("jarvis-ui-primary") ||
      defaultColors[getCurrentMode()].primary
  );

  const [backgroundColor, setBackgroundColor] = useState<string>(
    () =>
      localStorage.getItem("jarvis-ui-background") ||
      defaultColors[getCurrentMode()].background
  );

  const [textColor, setTextColor] = useState<string>(
    () =>
      localStorage.getItem("jarvis-ui-text") ||
      defaultColors[getCurrentMode()].text
  );

  // Apply theme class to document and handle system theme changes
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");

    if (theme === "system") {
      const systemTheme = window.matchMedia("(prefers-color-scheme: dark)")
        .matches
        ? "dark"
        : "light";

      root.classList.add(systemTheme);

      // Add listener for system theme changes
      const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

      // Define the handler function
      const handleSystemThemeChange = (e: MediaQueryListEvent) => {
        const newSystemTheme = e.matches ? "dark" : "light";
        root.classList.remove("light", "dark");
        root.classList.add(newSystemTheme);

        // Find equivalent colors for the new system theme
        resetColorsToThemeDefaults(newSystemTheme, true);
      };

      // Add the listener
      mediaQuery.addEventListener("change", handleSystemThemeChange);

      // Clean up
      return () => {
        mediaQuery.removeEventListener("change", handleSystemThemeChange);
      };
    } else {
      root.classList.add(theme);
    }
  }, [theme]);

  // Apply colors to CSS variables
  useEffect(() => {
    const root = window.document.documentElement;

    // Convert hex colors to HSL for CSS variables
    if (primaryColor) {
      try {
        const primaryHsl = hexToHsl(primaryColor);
        root.style.setProperty("--primary", primaryHsl);

        // Calculate foreground color based on luminance
        const luminance = parseInt(primaryHsl.split(" ")[2].replace("%", ""));
        const foreground = luminance > 60 ? "0 0% 10%" : "0 0% 98%";
        root.style.setProperty("--primary-foreground", foreground);

        // Add RGB values for use in rgba() functions
        const hexColor = primaryColor.replace("#", "");
        const r = parseInt(hexColor.substring(0, 2), 16);
        const g = parseInt(hexColor.substring(2, 4), 16);
        const b = parseInt(hexColor.substring(4, 6), 16);
        root.style.setProperty("--primary-rgb", `${r}, ${g}, ${b}`);

        console.log("Primary color set:", primaryHsl);
      } catch (e) {
        console.error("Error setting primary color:", e);
      }
    }

    if (accentColor) {
      try {
        const accentHsl = hexToHsl(accentColor);
        root.style.setProperty("--accent", accentHsl);

        // Calculate foreground color based on luminance
        const luminance = parseInt(accentHsl.split(" ")[2].replace("%", ""));
        const foreground = luminance > 60 ? "0 0% 10%" : "0 0% 98%";
        root.style.setProperty("--accent-foreground", foreground);

        console.log("Accent color set:", accentHsl);
      } catch (e) {
        console.error("Error setting accent color:", e);
      }
    }

    if (backgroundColor) {
      try {
        const backgroundHsl = hexToHsl(backgroundColor);
        root.style.setProperty("--background", backgroundHsl);

        // We'll use the text color directly instead of calculating from background
        console.log("Background color set:", backgroundHsl);
      } catch (e) {
        console.error("Error setting background color:", e);
      }
    }

    if (textColor) {
      try {
        const textHsl = hexToHsl(textColor);
        root.style.setProperty("--foreground", textHsl);

        // Set muted foreground as a slightly lighter version of text color
        const hslParts = textHsl.split(" ");
        const h = parseInt(hslParts[0]);
        const s = parseInt(hslParts[1]);
        const l = parseInt(hslParts[2]);

        // Adjust lightness for muted text (make it lighter or darker based on current lightness)
        const mutedL = l < 50 ? Math.min(l + 20, 90) : Math.max(l - 20, 30);
        root.style.setProperty("--muted-foreground", `${h} ${s}% ${mutedL}%`);

        console.log("Text color set:", textHsl);
      } catch (e) {
        console.error("Error setting text color:", e);
      }
    }
  }, [primaryColor, accentColor, backgroundColor, textColor]);

  // Store color preferences
  useEffect(() => {
    localStorage.setItem("jarvis-ui-accent", accentColor);
  }, [accentColor]);

  useEffect(() => {
    localStorage.setItem("jarvis-ui-primary", primaryColor);
  }, [primaryColor]);

  useEffect(() => {
    localStorage.setItem("jarvis-ui-background", backgroundColor);
  }, [backgroundColor]);

  useEffect(() => {
    localStorage.setItem("jarvis-ui-text", textColor);
  }, [textColor]);

  // Function to find equivalent color in new theme
  const findEquivalentColor = (
    currentColor: string,
    oldMode: "light" | "dark",
    newMode: "light" | "dark",
    colorType: "primary" | "accent" | "background" | "text"
  ): string => {
    try {
      // If the current color is exactly the default for the old mode, use the default for the new mode
      if (currentColor === defaultColors[oldMode][colorType]) {
        return defaultColors[newMode][colorType];
      }

      // Convert colors to HSL for comparison
      const currentHSL = hexToHSL(currentColor);
      const oldDefaultHSL = hexToHSL(defaultColors[oldMode][colorType]);
      const newDefaultHSL = hexToHSL(defaultColors[newMode][colorType]);

      // Calculate the relative difference between current color and old default
      const hueDiff = currentHSL.h - oldDefaultHSL.h;
      const satDiff = currentHSL.s - oldDefaultHSL.s;

      // For lightness, we need to handle it differently based on the theme direction
      let lightDiff;
      if (oldMode === "light" && newMode === "dark") {
        // Going from light to dark: invert the lightness difference
        lightDiff = 100 - currentHSL.l - (100 - oldDefaultHSL.l);
      } else if (oldMode === "dark" && newMode === "light") {
        // Going from dark to light: invert the lightness difference
        lightDiff = 100 - currentHSL.l - (100 - oldDefaultHSL.l);
      } else {
        // Same mode, just calculate the difference
        lightDiff = currentHSL.l - oldDefaultHSL.l;
      }

      // Apply the differences to the new default color
      let newH = (newDefaultHSL.h + hueDiff) % 360;
      if (newH < 0) newH += 360;

      let newS = newDefaultHSL.s + satDiff;
      newS = Math.max(0, Math.min(100, newS)); // Clamp between 0-100

      let newL = newDefaultHSL.l + lightDiff;
      newL = Math.max(0, Math.min(100, newL)); // Clamp between 0-100

      // Convert back to hex
      return hslToHex({ h: newH, s: newS, l: newL });
    } catch (e) {
      console.error("Error finding equivalent color:", e);
      return defaultColors[newMode][colorType]; // Fallback to default
    }
  };

  // Function to reset colors to theme defaults or find equivalents
  const resetColorsToThemeDefaults = (
    mode: "light" | "dark",
    findEquivalents = false
  ) => {
    const oldMode = mode === "light" ? "dark" : "light";

    if (
      findEquivalents &&
      localStorage.getItem("jarvis-ui-primary") &&
      localStorage.getItem("jarvis-ui-accent") &&
      localStorage.getItem("jarvis-ui-background") &&
      localStorage.getItem("jarvis-ui-text")
    ) {
      // Find equivalent colors in the new theme
      const newPrimary = findEquivalentColor(
        primaryColor,
        oldMode,
        mode,
        "primary"
      );
      const newAccent = findEquivalentColor(
        accentColor,
        oldMode,
        mode,
        "accent"
      );
      const newBackground = findEquivalentColor(
        backgroundColor,
        oldMode,
        mode,
        "background"
      );
      const newText = findEquivalentColor(textColor, oldMode, mode, "text");

      // Set the new equivalent colors
      setPrimaryColor(newPrimary);
      setAccentColor(newAccent);
      setBackgroundColor(newBackground);
      setTextColor(newText);

      // Update localStorage
      localStorage.setItem("jarvis-ui-primary", newPrimary);
      localStorage.setItem("jarvis-ui-accent", newAccent);
      localStorage.setItem("jarvis-ui-background", newBackground);
      localStorage.setItem("jarvis-ui-text", newText);
    } else {
      // Set colors to theme defaults
      setPrimaryColor(defaultColors[mode].primary);
      setAccentColor(defaultColors[mode].accent);
      setBackgroundColor(defaultColors[mode].background);
      setTextColor(defaultColors[mode].text);

      // Clear stored colors from localStorage
      localStorage.removeItem("jarvis-ui-primary");
      localStorage.removeItem("jarvis-ui-accent");
      localStorage.removeItem("jarvis-ui-background");
      localStorage.removeItem("jarvis-ui-text");
    }
  };

  const value = {
    theme,
    setTheme: (theme: Theme) => {
      localStorage.setItem(storageKey, theme);

      // Get the current mode before changing the theme
      const oldMode = getCurrentMode();

      // Update the theme state
      setTheme(theme);

      // Determine the new mode after theme change
      const newMode =
        theme === "system"
          ? window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark"
            : "light"
          : (theme as "light" | "dark");

      // If the mode has changed (light to dark or dark to light), find equivalent colors
      if (oldMode !== newMode) {
        resetColorsToThemeDefaults(newMode, true); // true = find equivalents
      }
    },

    // Add a function to reset colors to theme defaults
    resetColorsToDefaults: () => {
      resetColorsToThemeDefaults(getCurrentMode());
    },
    accentColor,
    setAccentColor: (color: string) => {
      setAccentColor(color);
    },
    primaryColor,
    setPrimaryColor: (color: string) => {
      setPrimaryColor(color);
    },
    backgroundColor,
    setBackgroundColor: (color: string) => {
      setBackgroundColor(color);
    },
    textColor,
    setTextColor: (color: string) => {
      setTextColor(color);
    },
  };

  return (
    <ThemeProviderContext.Provider {...props} value={value}>
      {children}
    </ThemeProviderContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeProviderContext);

  if (context === undefined)
    throw new Error("useTheme must be used within a ThemeProvider");

  return context;
};
