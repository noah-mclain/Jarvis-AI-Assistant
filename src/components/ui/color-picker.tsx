
import React from "react";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";

interface ColorPickerProps {
  color: string;
  onChange: (color: string) => void;
  label: string;
}

export function ColorPicker({ color, onChange, label }: ColorPickerProps) {
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
        <PopoverContent className="w-64 p-3">
          <div className="space-y-3">
            <h4 className="font-medium text-sm">{`Select ${label.toLowerCase()}`}</h4>
            <input
              type="color"
              value={color}
              onChange={(e) => onChange(e.target.value)}
              className="w-full h-8 cursor-pointer"
            />
            <div className="grid grid-cols-8 gap-2">
              {['#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#03a9f4', '#00bcd4', 
                '#009688', '#4caf50', '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722',
                '#795548', '#607d8b', '#000000', '#ffffff', '#6d28d9', '#db2777', '#ea580c', '#16a34a',
                '#0284c7', '#6b7280', '#78350f', '#047857'].map((presetColor) => (
                <button
                  key={presetColor}
                  onClick={() => onChange(presetColor)}
                  className={cn(
                    "w-5 h-5 rounded-full border border-border",
                    color === presetColor && "ring-2 ring-offset-2 ring-primary"
                  )}
                  style={{ backgroundColor: presetColor }}
                  aria-label={presetColor}
                />
              ))}
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
