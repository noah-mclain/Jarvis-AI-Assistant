/**
 * Model Buttons Component
 *
 * This component provides a set of buttons for different AI model functionalities:
 * - Speech-to-Text: For voice input and transcription
 * - Code Generation: For generating code based on prompts
 * - Text Generation/Conversation: For general text conversations
 * - NLP: For natural language processing tasks
 * - Story Generation: For creative writing and storytelling
 *
 * Each button has an appropriate icon and triggers a function to use the
 * corresponding model with a default prompt template.
 *
 * @module ModelButtons
 * @author Nada Mohamed
 */

import React from "react";
import { Button } from "@/components/ui/button";
import { Mic, Code, MessageSquare, Brain, BookText } from "lucide-react";
import { toast } from "@/components/ui/sonner";

/**
 * Props for the ModelButtons component
 *
 * @interface ModelButtonsProps
 * @property {function} onSelectModel - Callback function that receives the model type and default prompt
 */
interface ModelButtonsProps {
  onSelectModel: (modelType: string, defaultPrompt: string) => void;
}

/**
 * ModelButtons Component
 *
 * A component that displays buttons for different AI model functionalities.
 *
 * @param {ModelButtonsProps} props - Component props
 * @returns {JSX.Element} The rendered model buttons component
 */
export function ModelButtons({ onSelectModel }: ModelButtonsProps) {
  // Default prompts for each model type
  const defaultPrompts = {
    speechToText:
      "Transcribe the following audio file (provide a path to the audio file): ",
    codeGeneration: "Generate code for a function that: ",
    textGeneration: "Let's have a conversation about: ",
    nlp: "Analyze the sentiment, entities, and intent in this text: ",
    storyGeneration: "Write a creative story about: ",
  };

  /**
   * Handle button click for a specific model type
   *
   * @param {string} modelType - The type of model to use
   */
  const handleModelSelect = (modelType: string) => {
    // Get the default prompt for the selected model type
    const defaultPrompt =
      defaultPrompts[modelType as keyof typeof defaultPrompts];

    // Call the onSelectModel callback with the model type and default prompt
    onSelectModel(modelType, defaultPrompt);

    // Show a toast notification to indicate the model has been selected
    toast.success(`${modelType} model selected`, {
      description: "The input field has been updated with a default prompt.",
    });
  };

  return (
    <div className="flex flex-wrap justify-center gap-2 mb-2 p-1 rounded-lg bg-background/50 backdrop-blur-sm">
      {/* Speech-to-Text Button */}
      <Button
        variant="outline"
        size="icon"
        className="h-10 w-10 rounded-full transition-all duration-300 hover:scale-110"
        onClick={() => handleModelSelect("speechToText")}
        title="Speech-to-Text: Transcribe audio files to text"
      >
        <Mic className="h-4 w-4" />
        <span className="sr-only">Speech-to-Text</span>
      </Button>

      {/* Code Generation Button */}
      <Button
        variant="outline"
        size="icon"
        className="h-10 w-10 rounded-full transition-all duration-300 hover:scale-110"
        onClick={() => handleModelSelect("codeGeneration")}
        title="Code Generation: Generate code using DeepSeek Coder model"
      >
        <Code className="h-4 w-4" />
        <span className="sr-only">Code Generation</span>
      </Button>

      {/* Text Generation/Conversation Button */}
      <Button
        variant="outline"
        size="icon"
        className="h-10 w-10 rounded-full transition-all duration-300 hover:scale-110"
        onClick={() => handleModelSelect("textGeneration")}
        title="Text Generation: Generate text using FLAN-UL2 model"
      >
        <MessageSquare className="h-4 w-4" />
        <span className="sr-only">Text Generation/Conversation</span>
      </Button>

      {/* NLP Button */}
      <Button
        variant="outline"
        size="icon"
        className="h-10 w-10 rounded-full transition-all duration-300 hover:scale-110"
        onClick={() => handleModelSelect("nlp")}
        title="Natural Language Processing: Analyze sentiment, entities, and intent"
      >
        <Brain className="h-4 w-4" />
        <span className="sr-only">Natural Language Processing</span>
      </Button>

      {/* Story Generation Button */}
      <Button
        variant="outline"
        size="icon"
        className="h-10 w-10 rounded-full transition-all duration-300 hover:scale-110"
        onClick={() => handleModelSelect("storyGeneration")}
        title="Story Generation: Create creative stories with context-aware generation"
      >
        <BookText className="h-4 w-4" />
        <span className="sr-only">Story Generation</span>
      </Button>
    </div>
  );
}
