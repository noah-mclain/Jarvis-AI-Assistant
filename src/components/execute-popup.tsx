/**
 * Execute Popup Component
 *
 * This component provides a popup window with a text box and two buttons:
 * - Execute: For executing commands with a 30-second wait time
 * - Speech-to-Text: For voice input with a 30-second wait time
 *
 * The component is designed to work with the main chat interface.
 *
 * @module ExecutePopup
 * @author Nada Mohamed
 */

import React, { useState, useRef, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { TextareaAutosize } from "@/components/ui/textarea-autosize";
import { Play, Mic, Terminal, X } from "lucide-react";
import { toast } from "@/components/ui/sonner";

/**
 * Props for the ExecutePopup component
 *
 * @interface ExecutePopupProps
 * @property {function} onExecute - Callback function that receives the command text when executed
 */
interface ExecutePopupProps {
  onExecute?: (command: string) => void;
}

/**
 * ExecutePopup Component
 *
 * A dialog component that provides a text input and buttons for executing commands
 * or using speech-to-text with a 30-second wait time.
 *
 * @param {ExecutePopupProps} props - Component props
 * @returns {JSX.Element} The rendered execute popup component
 */
export function ExecutePopup({ onExecute }: ExecutePopupProps) {
  // State to track the current command input
  const [command, setCommand] = useState("");

  // State to track if execution is in progress
  const [isExecuting, setIsExecuting] = useState(false);

  // State to track the countdown timer
  const [countdown, setCountdown] = useState(30);

  // State to track if the dialog is open
  const [isOpen, setIsOpen] = useState(false);

  // State to track if speech-to-text is active
  const [isSpeechToText, setIsSpeechToText] = useState(false);

  // Reference to the countdown interval
  const countdownRef = useRef<NodeJS.Timeout | null>(null);

  /**
   * Handles the execution of a command with a 5-second wait time
   */
  const handleExecute = () => {
    if (!command.trim()) {
      toast.error("Please enter a command to execute");
      return;
    }

    setIsExecuting(true);
    setIsSpeechToText(false);
    setCountdown(5); // 5 seconds for written execute

    // Start the countdown
    countdownRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          // Clear the interval when countdown reaches 0
          if (countdownRef.current) {
            clearInterval(countdownRef.current);
          }

          // Execute the command by sending it to the backend API
          executeCommand(command);

          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  /**
   * Sends the command to the backend API for execution
   *
   * @param {string} commandText - The command text to execute
   */
  const executeCommand = async (commandText: string) => {
    try {
      // Send the command to the backend API
      const response = await fetch("/api/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          command: commandText,
          // You can specify a model type here if needed
          // modelType: 'codeGeneration',
        }),
      });

      if (response.ok) {
        // Process successful response
        const data = await response.json();

        // Show success toast with the result
        toast.success("Command executed", {
          description:
            data.result.substring(0, 100) +
            (data.result.length > 100 ? "..." : ""),
        });

        // Call the onExecute callback if provided
        if (onExecute) {
          onExecute(commandText);
        }
      } else {
        // Handle API error
        const errorData = await response.json().catch(() => ({}));
        toast.error("Failed to execute command", {
          description: errorData.error || response.statusText,
        });
      }
    } catch (error) {
      // Handle network or server errors
      console.error("Error executing command:", error);
      toast.error(
        "Cannot connect to the server. The application may need to be restarted."
      );
    } finally {
      // Reset states
      setIsExecuting(false);
      setIsSpeechToText(false);
      setCommand("");
      setIsOpen(false);
    }
  };

  /**
   * Handles the speech-to-text functionality with a 30-second wait time
   */
  const handleSpeechToText = () => {
    setIsExecuting(true);
    setIsSpeechToText(true);
    setCountdown(30); // 30 seconds for speech-to-text

    // Start the countdown
    countdownRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          // Clear the interval when countdown reaches 0
          if (countdownRef.current) {
            clearInterval(countdownRef.current);
          }

          // In a real implementation, this would capture audio and send it for transcription
          // For now, we'll simulate a transcription result
          const simulatedTranscription =
            "Show me the weather forecast for today";
          setCommand(simulatedTranscription);

          // Process the transcribed text
          processSpeechToText(simulatedTranscription);

          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  /**
   * Processes the transcribed text by sending it to the backend API
   *
   * @param {string} text - The transcribed text to process
   */
  const processSpeechToText = async (text: string) => {
    try {
      // Send the transcribed text to the backend API
      const response = await fetch("/api/speech-to-text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text,
          modelType: "speechToText",
        }),
      });

      if (response.ok) {
        // Process successful response
        const data = await response.json();

        // Show success toast with the result
        toast.success("Speech processed", {
          description:
            data.result.substring(0, 100) +
            (data.result.length > 100 ? "..." : ""),
        });

        // Call the onExecute callback if provided
        if (onExecute) {
          onExecute(text);
        }
      } else {
        // Handle API error
        const errorData = await response.json().catch(() => ({}));
        toast.error("Failed to process speech", {
          description: errorData.error || response.statusText,
        });
      }
    } catch (error) {
      // Handle network or server errors
      console.error("Error processing speech:", error);
      toast.error(
        "Cannot connect to the server. The application may need to be restarted."
      );
    } finally {
      // Reset states
      setIsExecuting(false);
      setIsSpeechToText(false);
      setIsOpen(false);
    }
  };

  /**
   * Cancels the execution and clears the countdown
   */
  const handleCancel = () => {
    if (countdownRef.current) {
      clearInterval(countdownRef.current);
    }
    setIsExecuting(false);
    setIsSpeechToText(false);
    setCountdown(30);
  };

  /**
   * Clean up the interval when the component unmounts
   */
  useEffect(() => {
    return () => {
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, []);

  /**
   * Clean up the interval when the dialog closes
   */
  useEffect(() => {
    if (!isOpen) {
      handleCancel();
    }
  }, [isOpen]);

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className="h-10 w-10 rounded-full transition-all duration-300 hover:scale-110"
          title="Command Terminal: Execute commands (5-second delay) or use speech-to-text (30-second listening period)"
        >
          <Terminal className="h-5 w-5" />
          <span className="sr-only">Command Terminal</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Command Terminal</DialogTitle>
          <DialogDescription>
            Enter a command to execute or use speech-to-text.
            {isExecuting && (
              <div className="mt-2 text-primary font-medium">
                {isSpeechToText ? (
                  <div>
                    <div>Waiting to hear command...</div>
                    <div>Time remaining: {countdown} seconds</div>
                  </div>
                ) : (
                  <div>Executing in {countdown} seconds...</div>
                )}
              </div>
            )}
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col space-y-4 py-4">
          <TextareaAutosize
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            placeholder="Enter command here..."
            className="min-h-[100px] border rounded-md p-3 focus-visible:ring-1 focus-visible:ring-ring scrollbar-themed"
            disabled={isExecuting}
            maxRows={8}
          />
        </div>
        <DialogFooter className="flex justify-between sm:justify-between">
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={handleSpeechToText}
              disabled={isExecuting}
              className="gap-2 transition-all duration-300 hover:scale-105"
              title="Convert speech to text with a 30-second safety delay"
            >
              <Mic className="h-4 w-4" />
              Speech to Text
            </Button>
            <Button
              type="button"
              onClick={handleExecute}
              disabled={isExecuting || !command.trim()}
              className="gap-2 transition-all duration-300 hover:scale-105"
              title="Execute the command with a 5-second safety delay"
            >
              <Play className="h-4 w-4" />
              Execute
            </Button>
          </div>
          {isExecuting && (
            <Button
              type="button"
              variant="destructive"
              onClick={handleCancel}
              className="gap-2 transition-all duration-300 hover:scale-105"
              title="Cancel the execution"
            >
              <X className="h-4 w-4" />
              Cancel
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
