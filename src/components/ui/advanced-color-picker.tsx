import React, { useState, useEffect, useRef } from "react";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface HSLColor {
  h: number;
  s: number;
  l: number;
}

// Convert hex to HSL object
function hexToHSL(hex: string): HSLColor {
  // Remove the # if it exists
  hex = hex.replace(/^#/, "");

  // Parse the hex values
  let r = parseInt(hex.substring(0, 2), 16) / 255;
  let g = parseInt(hex.substring(2, 4), 16) / 255;
  let b = parseInt(hex.substring(4, 6), 16) / 255;

  // Find the min and max values to compute the luminance
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0;
  let s = 0;
  let l = (max + min) / 2;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      case b:
        h = (r - g) / d + 4;
        break;
    }

    h = h / 6;
  }

  // Convert to degrees, percentage, percentage format
  h = Math.round(h * 360);
  s = Math.round(s * 100);
  l = Math.round(l * 100);

  return { h, s, l };
}

// Convert HSL object to hex
function hslToHex({ h, s, l }: HSLColor): string {
  l /= 100;
  const a = (s * Math.min(l, 1 - l)) / 100;
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * color)
      .toString(16)
      .padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

// Convert HSL object to CSS string
function hslToCssString({ h, s, l }: HSLColor): string {
  return `${h} ${s}% ${l}%`;
}

// Color wheel component
function ColorWheel({
  hslColor,
  onChange,
}: {
  hslColor: HSLColor;
  onChange: (h: number, s: number) => void;
}) {
  const wheelRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Calculate position from HSL with improved accuracy
  const getPositionFromHSL = (h: number, s: number) => {
    // Convert hue to radians, adjusting for CSS conic gradient which starts at top (270 degrees)
    const angleRad = ((h + 270) % 360) * (Math.PI / 180);

    // Scale saturation to wheel radius (0-100)
    const radius = (s / 100) * 100;

    // Calculate x and y coordinates
    const x = radius * Math.cos(angleRad);
    const y = radius * Math.sin(angleRad);

    return { x, y };
  };

  // Calculate HSL from position with improved accuracy
  const getHSLFromPosition = (x: number, y: number, wheelSize: number) => {
    const centerX = wheelSize / 2;
    const centerY = wheelSize / 2;

    // Calculate relative position from center
    const relX = x - centerX;
    const relY = y - centerY;

    // Calculate angle in degrees, adjusting for CSS conic gradient which starts at top
    let angle = Math.atan2(relY, relX) * (180 / Math.PI);
    // Convert to 0-360 range and adjust for CSS conic gradient starting point
    let h = (angle + 90) % 360;
    if (h < 0) h += 360;

    // Calculate distance from center (saturation)
    const distance = Math.sqrt(relX * relX + relY * relY);
    const maxRadius = wheelSize / 2;

    // Calculate saturation as percentage of maximum radius
    let s = (distance / maxRadius) * 100;

    // Ensure saturation is within bounds
    s = Math.max(0, Math.min(100, s));

    return { h, s };
  };

  // Handle mouse/touch events
  const handlePointerDown = (e: React.PointerEvent) => {
    if (wheelRef.current) {
      setIsDragging(true);
      wheelRef.current.setPointerCapture(e.pointerId);
      updateColorFromEvent(e);
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (isDragging) {
      updateColorFromEvent(e);
    }
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (wheelRef.current) {
      setIsDragging(false);
      wheelRef.current.releasePointerCapture(e.pointerId);
    }
  };

  const updateColorFromEvent = (e: React.PointerEvent) => {
    if (wheelRef.current) {
      const rect = wheelRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const wheelSize = rect.width;

      const { h, s } = getHSLFromPosition(x, y, wheelSize);
      onChange(Math.round(h), Math.round(s));
    }
  };

  // Calculate selector position
  const { x: selectorX, y: selectorY } = getPositionFromHSL(
    hslColor.h,
    hslColor.s
  );
  const wheelSize = 200; // Size of the wheel
  const selectorSize = 12; // Size of the selector dot

  // Generate wheel background with CSS conic gradient - improved for better color accuracy
  const generateWheelBackground = () => {
    const steps = 72; // Increased number of color steps for smoother gradient
    const colors = [];

    // Start from the top (270 degrees in CSS conic gradient)
    for (let i = 0; i <= steps; i++) {
      const h = (i / steps) * 360;
      colors.push(`hsl(${h}, 100%, 50%) ${(i / steps) * 100}%`);
    }

    // Add the first color again to close the circle
    colors.push(`hsl(0, 100%, 50%) 100%`);

    // Create a radial gradient for saturation and a conic gradient for hue
    return `
      conic-gradient(from 0deg, ${colors.join(", ")}),
      radial-gradient(circle at center, white 0%, transparent 70%)
    `;
  };

  return (
    <div className="relative mb-4">
      <div
        ref={wheelRef}
        className="w-[200px] h-[200px] rounded-full mx-auto relative overflow-hidden"
        style={{
          background: generateWheelBackground(),
          boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.1)",
        }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        {/* Saturation overlay - white to transparent radial gradient */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background:
              "radial-gradient(circle at center, white 0%, transparent 100%)",
            mixBlendMode: "overlay",
          }}
        />
        {/* Selector dot */}
        <div
          className="absolute rounded-full border-2 border-white shadow-md"
          style={{
            width: `${selectorSize}px`,
            height: `${selectorSize}px`,
            backgroundColor: `hsl(${hslColor.h}, ${hslColor.s}%, ${hslColor.l}%)`,
            transform: `translate(${
              selectorX + wheelSize / 2 - selectorSize / 2
            }px, ${selectorY + wheelSize / 2 - selectorSize / 2}px)`,
            pointerEvents: "none",
          }}
        />
      </div>
    </div>
  );
}

interface AdvancedColorPickerProps {
  color: string;
  onChange: (color: string) => void;
  label: string;
}

export function AdvancedColorPicker({
  color,
  onChange,
  label,
}: AdvancedColorPickerProps) {
  const [hslColor, setHslColor] = useState<HSLColor>(hexToHSL(color));
  const [hexValue, setHexValue] = useState<string>(color);
  const [activeTab, setActiveTab] = useState<string>("wheel");

  // Update HSL and hex when color prop changes
  useEffect(() => {
    setHslColor(hexToHSL(color));
    setHexValue(color);
  }, [color]);

  // Handle HSL changes
  const handleHSLChange = (property: keyof HSLColor, value: number) => {
    const newHslColor = { ...hslColor, [property]: value };
    setHslColor(newHslColor);
    const newHexColor = hslToHex(newHslColor);
    setHexValue(newHexColor);
    onChange(newHexColor);
  };

  // Handle wheel changes (hue and saturation)
  const handleWheelChange = (h: number, s: number) => {
    const newHslColor = { ...hslColor, h, s };
    setHslColor(newHslColor);
    const newHexColor = hslToHex(newHslColor);
    setHexValue(newHexColor);
    onChange(newHexColor);
  };

  // Handle hex input change
  const handleHexChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newHex = e.target.value;
    setHexValue(newHex);

    // Only update if it's a valid hex color
    if (/^#[0-9A-F]{6}$/i.test(newHex)) {
      setHslColor(hexToHSL(newHex));
      onChange(newHex);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <Label>{label}</Label>
      <Popover>
        <PopoverTrigger asChild>
          <button
            className={cn(
              "w-10 h-10 rounded-full border-2 border-border",
              "hover:shadow-md transition-shadow"
            )}
            style={{ backgroundColor: color }}
            aria-label={`Select ${label.toLowerCase()}`}
          />
        </PopoverTrigger>
        <PopoverContent className="w-80 p-4">
          <Tabs
            defaultValue="hex"
            value={activeTab}
            onValueChange={setActiveTab}
          >
            <TabsList className="grid w-full grid-cols-3 mb-4">
              <TabsTrigger value="wheel">Wheel</TabsTrigger>
              <TabsTrigger value="hex">Hex</TabsTrigger>
              <TabsTrigger value="hsl">HSL</TabsTrigger>
            </TabsList>

            <TabsContent value="wheel" className="space-y-4">
              <ColorWheel hslColor={hslColor} onChange={handleWheelChange} />

              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Lightness ({hslColor.l}%)</Label>
                </div>
                <Slider
                  value={[hslColor.l]}
                  min={0}
                  max={100}
                  step={1}
                  onValueChange={(value) => handleHSLChange("l", value[0])}
                />
              </div>
            </TabsContent>

            <TabsContent value="hex" className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="hex-color">Hex Color</Label>
                <Input
                  id="hex-color"
                  value={hexValue}
                  onChange={handleHexChange}
                  placeholder="#000000"
                />
              </div>

              <div className="grid grid-cols-8 gap-2 mt-4">
                {[
                  "#f44336",
                  "#e91e63",
                  "#9c27b0",
                  "#673ab7",
                  "#3f51b5",
                  "#2196f3",
                  "#03a9f4",
                  "#00bcd4",
                  "#009688",
                  "#4caf50",
                  "#8bc34a",
                  "#cddc39",
                  "#ffeb3b",
                  "#ffc107",
                  "#ff9800",
                  "#ff5722",
                  "#795548",
                  "#607d8b",
                  "#000000",
                  "#ffffff",
                  "#6d28d9",
                  "#db2777",
                  "#ea580c",
                  "#16a34a",
                  "#0284c7",
                  "#6b7280",
                  "#78350f",
                  "#047857",
                ].map((presetColor) => (
                  <button
                    key={presetColor}
                    onClick={() => {
                      setHexValue(presetColor);
                      setHslColor(hexToHSL(presetColor));
                      onChange(presetColor);
                    }}
                    className={cn(
                      "w-6 h-6 rounded-full border border-border",
                      color.toLowerCase() === presetColor.toLowerCase() &&
                        "ring-2 ring-offset-2 ring-primary"
                    )}
                    style={{ backgroundColor: presetColor }}
                    aria-label={presetColor}
                  />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="hsl" className="space-y-4">
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Hue ({hslColor.h}°)</Label>
                  </div>
                  <Slider
                    value={[hslColor.h]}
                    min={0}
                    max={360}
                    step={1}
                    onValueChange={(value) => handleHSLChange("h", value[0])}
                    className="hue-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Saturation ({hslColor.s}%)</Label>
                  </div>
                  <Slider
                    value={[hslColor.s]}
                    min={0}
                    max={100}
                    step={1}
                    onValueChange={(value) => handleHSLChange("s", value[0])}
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Lightness ({hslColor.l}%)</Label>
                  </div>
                  <Slider
                    value={[hslColor.l]}
                    min={0}
                    max={100}
                    step={1}
                    onValueChange={(value) => handleHSLChange("l", value[0])}
                  />
                </div>
              </div>
            </TabsContent>
          </Tabs>

          <div className="mt-4 pt-4 border-t">
            <div
              className="w-full h-10 rounded-md"
              style={{ backgroundColor: color }}
            />
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
