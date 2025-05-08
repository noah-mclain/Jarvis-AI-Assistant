import { useTheme } from "@/components/theme-provider";
import { useFontSize } from "@/components/font-size-provider";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Settings, Plus, Minus, RotateCcw } from "lucide-react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { AdvancedColorPicker } from "@/components/ui/advanced-color-picker";
import { InfoTooltip } from "@/components/ui/info-tooltip";
import { TooltipProvider } from "@/components/ui/tooltip";

export function SettingsPanel() {
  const {
    theme,
    setTheme,
    accentColor,
    setAccentColor,
    primaryColor,
    setPrimaryColor,
    backgroundColor,
    setBackgroundColor,
    textColor,
    setTextColor,
    resetColorsToDefaults,
  } = useTheme();

  const { fontSize, increaseFontSize, decreaseFontSize, resetFontSize } =
    useFontSize();

  return (
    <TooltipProvider>
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="rounded-full">
            <Settings className="h-[1.2rem] w-[1.2rem]" />
            <span className="sr-only">Settings</span>
          </Button>
        </SheetTrigger>
        <SheetContent className="w-full sm:max-w-md">
          <SheetHeader>
            <SheetTitle className="sheet-title">Settings</SheetTitle>
            <SheetDescription className="sheet-description">
              Customize your Jarvis AI experience
            </SheetDescription>
          </SheetHeader>
          <div className="py-6 space-y-8">
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <h3 className="font-medium settings-section-title">Theme</h3>
                <InfoTooltip
                  content={
                    <div>
                      <p>Choose between light, dark, or system theme.</p>
                      <p className="mt-2">
                        System theme will automatically match your device's
                        theme settings.
                      </p>
                    </div>
                  }
                  iconSize={14}
                />
              </div>
              <RadioGroup
                value={theme}
                onValueChange={(value) =>
                  setTheme(value as "light" | "dark" | "system")
                }
                className="flex gap-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="light" id="theme-light" />
                  <Label htmlFor="theme-light">Light</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="dark" id="theme-dark" />
                  <Label htmlFor="theme-dark">Dark</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="system" id="theme-system" />
                  <Label htmlFor="theme-system">System</Label>
                </div>
              </RadioGroup>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <h3 className="font-medium settings-section-title">
                    Font Size
                  </h3>
                  <InfoTooltip
                    content={
                      <div>
                        <p>
                          Font size adjusts automatically based on window size
                          and can be manually adjusted between 12px and 24px.
                        </p>
                        <p className="mt-2">
                          Use the buttons below or keyboard shortcuts (Ctrl/Cmd
                          + / -) to adjust the font size.
                        </p>
                      </div>
                    }
                    iconSize={14}
                  />
                </div>
                <div className="text-sm text-muted-foreground">
                  Current: {fontSize}px
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={decreaseFontSize}
                    className="h-8 w-8"
                  >
                    <Minus className="h-4 w-4" />
                    <span className="sr-only">Decrease font size</span>
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={increaseFontSize}
                    className="h-8 w-8"
                  >
                    <Plus className="h-4 w-4" />
                    <span className="sr-only">Increase font size</span>
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={resetFontSize}
                    className="h-8 w-8"
                  >
                    <RotateCcw className="h-4 w-4" />
                    <span className="sr-only">Reset font size</span>
                  </Button>
                </div>
                <div className="text-xs text-muted-foreground">
                  <span className="hidden sm:inline">Keyboard shortcuts: </span>
                  <kbd className="px-1 bg-muted rounded text-xs">
                    {navigator.platform.indexOf("Mac") > -1 ? "Cmd" : "Ctrl"}+/-
                  </kbd>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <h3 className="font-medium settings-section-title">
                    Color Customization
                  </h3>
                  <InfoTooltip
                    content={
                      <div>
                        <p>
                          Use the color wheel for intuitive selection or the HSL
                          tab for fine-tuning saturation, hue, and lightness.
                        </p>
                        <p className="mt-2">
                          When switching between light and dark themes, your
                          custom colors will be intelligently adapted to
                          maintain a similar feel in the new theme.
                        </p>
                      </div>
                    }
                    iconSize={14}
                  />
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={resetColorsToDefaults}
                  className="text-xs"
                >
                  Reset to Default Colors
                </Button>
              </div>
              <div className="flex flex-col sm:flex-row flex-wrap gap-6">
                <AdvancedColorPicker
                  label="Primary Color"
                  color={primaryColor}
                  onChange={setPrimaryColor}
                />
                <AdvancedColorPicker
                  label="Accent Color"
                  color={accentColor}
                  onChange={setAccentColor}
                />
                <AdvancedColorPicker
                  label="Background Color"
                  color={backgroundColor}
                  onChange={setBackgroundColor}
                />
                <AdvancedColorPicker
                  label="Text Color"
                  color={textColor}
                  onChange={setTextColor}
                />
              </div>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </TooltipProvider>
  );
}
