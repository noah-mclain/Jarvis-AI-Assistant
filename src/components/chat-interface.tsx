/**
 * Chat Interface Component
 *
 * This is the main component for the Jarvis AI Assistant chat interface.
 * It manages the state and interactions between the user and the AI assistant,
 * including:
 * - Chat history management
 * - Message sending and receiving
 * - UI layout and responsiveness
 * - Offline mode handling
 *
 * The component is designed to work fully offline, communicating with a local
 * Flask server running on the same machine. No internet connection is required.
 *
 * @module ChatInterface
 * @author Nada Mohamed
 */

import { useState, useEffect, useCallback } from "react";
import { ChatSidebar } from "@/components/chat-sidebar";
import { ChatMessage } from "@/components/chat-message";
import { ChatInput } from "@/components/chat-input";
import { ThemeToggle } from "@/components/theme-toggle";
import { SettingsPanel } from "@/components/settings-panel";
import { Message } from "@/types/message";
import { toast } from "@/components/ui/sonner";
import { useWindowSize } from "@/hooks/use-window-size";
import { useIsMobile } from "@/hooks/use-mobile";

/**
 * Chat object type definition
 *
 * @typedef {Object} Chat
 * @property {string} id - Unique identifier for the chat
 * @property {string} title - Display title for the chat
 * @property {string} date - Formatted date when the chat was created
 */
type Chat = {
  id: string;
  title: string;
  date: string;
};

/**
 * ChatInterface Component
 *
 * Main component that handles the chat interface for the Jarvis AI Assistant.
 * This component manages the state of chats, messages, and user interactions.
 *
 * @returns {JSX.Element} The rendered chat interface
 */
export function ChatInterface() {
  // State for tracking the currently active chat
  const [activeChat, setActiveChat] = useState<string>("");

  // State for storing all available chats
  const [chats, setChats] = useState<Chat[]>([]);

  // State for storing messages in the current active chat
  const [messages, setMessages] = useState<Message[]>([]);

  // State for tracking loading states during API calls
  const [isLoading, setIsLoading] = useState(false);

  // State to track if the component has completed initial data loading
  const [isInitialized, setIsInitialized] = useState(false);

  // State for tracking offline status (always false in this offline application)
  const [isOffline, setIsOffline] = useState(false);

  // State for tracking the currently selected model type
  const [currentModelType, setCurrentModelType] = useState<string>("");

  // Custom hooks for responsive design
  const windowSize = useWindowSize(); // Tracks window dimensions
  const isMobile = useIsMobile(); // Determines if viewport is mobile size

  /**
   * Offline Mode Management
   *
   * Since this application runs fully offline (connecting to a local server),
   * we don't need to check for internet connectivity. The application is
   * designed to work without an internet connection.
   *
   * This effect initializes the application in "online" mode since we're
   * connecting to a local server running on the same machine.
   */
  useEffect(() => {
    // Always set offline status to false since we're running locally
    setIsOffline(false);
  }, []);

  /**
   * Initial Data Loading
   *
   * This effect runs once on component mount and handles:
   * 1. Fetching all available chats from the local server
   * 2. Determining which chat to show initially
   * 3. Creating a new chat if needed
   *
   * The logic prioritizes:
   * - Using an existing empty chat if available
   * - Creating a new chat if all existing chats have messages
   * - Handling error cases gracefully
   */
  useEffect(() => {
    /**
     * Fetches all chats from the server and initializes the active chat
     */
    const fetchChats = async () => {
      try {
        // Fetch all available chats from the API
        const response = await fetch("/api/chats");

        if (response.ok) {
          const data = await response.json();
          setChats(data);

          // Only initialize a chat if no chat is currently active
          if (!activeChat) {
            // Check if there are any existing chats
            if (data.length > 0) {
              // Get the most recent chat (last in the array)
              const mostRecentChat = data[data.length - 1];

              // Check if the most recent chat has any messages beyond the welcome message
              const messagesResponse = await fetch(
                `/api/chats/${mostRecentChat.id}/messages`
              );

              if (messagesResponse.ok) {
                const messagesData = await messagesResponse.json();

                // Check if there are only system/welcome messages or no messages
                const hasOnlyWelcomeMessage =
                  messagesData.length === 0 ||
                  (messagesData.length === 1 &&
                    messagesData[0].role === "assistant" &&
                    messagesData[0].id === "welcome");

                if (hasOnlyWelcomeMessage) {
                  // If the most recent chat has only the welcome message, use it
                  // This avoids creating unnecessary new chats
                  setActiveChat(mostRecentChat.id);
                  setMessages(messagesData);
                  console.log("Using existing empty chat:", mostRecentChat.id);
                } else if (
                  messagesData.length > 1 ||
                  (messagesData.length === 1 && messagesData[0].role === "user")
                ) {
                  // If the most recent chat has user messages, create a new one
                  // This ensures each new session starts with a clean chat
                  await createNewChat();
                  console.log(
                    "Most recent chat has user messages, creating new chat"
                  );
                } else {
                  // Use the existing chat for any other case
                  setActiveChat(mostRecentChat.id);
                  setMessages(messagesData);
                  console.log("Using existing chat:", mostRecentChat.id);
                }
              } else {
                // If we can't fetch messages, create a new chat to be safe
                // This handles API errors gracefully
                await createNewChat();
              }
            } else {
              // No existing chats, create a new one
              // This handles first-time use of the application
              await createNewChat();
              console.log("No existing chats, creating new chat");
            }
          }
        } else {
          // Handle API error when fetching chats
          const errorData = await response.json().catch(() => ({}));
          console.error(
            "Failed to fetch chats:",
            errorData.error || response.statusText
          );
          toast.error("Failed to load chats. Please try again.");
        }
      } catch (error) {
        // Handle network or server errors
        console.error("Error fetching chats:", error);
        toast.error(
          "Cannot connect to the local server. The application may need to be restarted."
        );
      } finally {
        // Mark initialization as complete regardless of success/failure
        setIsInitialized(true);
      }
    };

    // Execute the fetch function
    fetchChats();
  }, []); // Empty dependency array ensures this runs only once on mount

  /**
   * Creates a new chat via the API
   *
   * This function:
   * 1. Makes a POST request to create a new chat
   * 2. Adds the new chat to the chats state
   * 3. Sets the new chat as active
   * 4. Clears the messages state
   *
   * @returns {Promise<void>}
   */
  const createNewChat = async () => {
    // Don't create a new chat if offline
    if (isOffline) {
      return;
    }

    try {
      // Make API request to create a new chat
      const response = await fetch("/api/chats", {
        method: "POST",
      });

      if (response.ok) {
        // Process successful response
        const newChat = await response.json();

        // Add the new chat to the list of chats
        setChats((prev) => [...prev, newChat]);

        // Set the new chat as the active chat
        setActiveChat(newChat.id);

        // Clear the messages state for the new chat
        setMessages([]);
      } else {
        // Handle API error
        const errorData = await response.json().catch(() => ({}));
        console.error(
          "Failed to create new chat:",
          errorData.error || response.statusText
        );
      }
    } catch (error) {
      // Handle network or server errors
      console.error("Error creating new chat:", error);
    }
  };

  /**
   * Message Loading Effect
   *
   * This effect runs whenever the active chat changes or when initialization completes.
   * It fetches all messages for the currently active chat from the server.
   *
   * Dependencies:
   * - activeChat: Triggers when the user switches to a different chat
   * - isInitialized: Ensures we don't fetch before initial setup is complete
   */
  useEffect(() => {
    // Only fetch messages if we have an active chat and initialization is complete
    if (activeChat && isInitialized) {
      /**
       * Fetches messages for the active chat
       */
      const fetchMessages = async () => {
        // Show loading indicator
        setIsLoading(true);

        try {
          // Fetch messages for the active chat
          const response = await fetch(`/api/chats/${activeChat}/messages`);

          if (response.ok) {
            // Process successful response
            const data = await response.json();
            setMessages(data);
          } else {
            // Handle API error
            const errorData = await response.json().catch(() => ({}));
            console.error(
              "Failed to fetch messages:",
              errorData.error || response.statusText
            );
            toast.error("Failed to load messages. Please try again.");
          }
        } catch (error) {
          // Handle network or server errors
          console.error("Error fetching messages:", error);
          toast.error(
            "Cannot connect to the local server. The application may need to be restarted."
          );
        } finally {
          // Hide loading indicator regardless of success/failure
          setIsLoading(false);
        }
      };

      // Execute the fetch function
      fetchMessages();
    }
  }, [activeChat, isInitialized]);

  /**
   * Handles sending a user message to the AI assistant
   *
   * This function:
   * 1. Validates the active chat and offline status
   * 2. Adds the user message to the UI immediately for better UX
   * 3. Sends the message to the backend API
   * 4. Processes the AI response
   * 5. Handles errors gracefully
   *
   * @param {string} message - The message text to send
   * @returns {Promise<void>}
   */
  const handleSendMessage = async (message: string) => {
    // Ensure we have an active chat to send the message to
    if (!activeChat) {
      return;
    }

    // Don't send messages if offline
    if (isOffline) {
      toast.error("You are offline. Cannot send messages.");
      return;
    }

    // Create a temporary user message object to show immediately
    // This provides instant feedback to the user before the API responds
    const userMessage: Message = {
      id: Date.now().toString(), // Temporary ID using timestamp
      role: "user", // Message is from the user
      content: message, // The actual message text
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    // Add the user message to the UI immediately
    setMessages((prev) => [...prev, userMessage]);

    // Show loading indicator while waiting for AI response
    setIsLoading(true);

    try {
      // Send message to backend API with the current model type
      const response = await fetch(`/api/chats/${activeChat}/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: message,
          modelType: currentModelType,
        }),
      });

      if (response.ok) {
        // Process successful response
        const data = await response.json();

        // Update chat title if this is the first message in a new chat
        if (data.chatTitleUpdated) {
          setChats((prevChats) =>
            prevChats.map((chat) =>
              chat.id === activeChat ? { ...chat, title: data.newTitle } : chat
            )
          );
        }

        // Replace the temporary user message and add the AI response
        // This ensures we have the correct message IDs from the server
        setMessages((prev) => [
          ...prev.filter((msg) => msg.id !== userMessage.id), // Remove temporary message
          data.userMessage, // Add server-generated user message
          data.aiResponse, // Add AI response
        ]);
      } else {
        // Handle API error
        const errorData = await response.json().catch(() => ({}));
        console.error(
          "Failed to send message:",
          errorData.error || response.statusText
        );
        toast.error("Failed to send message. Please try again.");

        // Remove the temporary user message since the request failed
        setMessages((prev) => prev.filter((msg) => msg.id !== userMessage.id));
      }
    } catch (error) {
      // Handle network or server errors
      console.error("Error sending message:", error);
      toast.error(
        "Cannot connect to the local server. The application may need to be restarted."
      );

      // Remove the temporary user message since the request failed
      setMessages((prev) => prev.filter((msg) => msg.id !== userMessage.id));
    } finally {
      // Hide loading indicator regardless of success/failure
      setIsLoading(false);
    }
  };

  /**
   * Handles the "New Chat" action
   *
   * This function:
   * 1. Checks if the current chat is already empty (to avoid creating unnecessary chats)
   * 2. Looks for existing empty chats that can be reused
   * 3. Creates a new chat only if necessary
   *
   * This approach optimizes resource usage by reusing empty chats
   * rather than creating new ones unnecessarily.
   *
   * @returns {Promise<void>}
   */
  const handleNewChat = async () => {
    // Don't create a new chat if offline
    if (isOffline) {
      toast.error("You are offline. Cannot create new chat.");
      return;
    }

    // Show loading indicator
    setIsLoading(true);

    try {
      // Check if current chat is empty or has only the welcome message
      // We don't need to create a new chat if the current one is already empty
      const hasOnlyWelcomeMessage =
        messages.length === 0 ||
        (messages.length === 1 &&
          messages[0].role === "assistant" &&
          messages[0].id === "welcome");

      if (activeChat && hasOnlyWelcomeMessage) {
        // Current chat is already empty or has only the welcome message, just keep using it
        // This avoids creating unnecessary new chats
        console.log(
          "Current chat is already empty or has only the welcome message, no need to create a new one"
        );
        setIsLoading(false);
        return;
      }

      // Check if there's already an empty chat in the list that we can reuse
      let emptyChat = null;

      // Filter potential empty chats (with title "New Conversation")
      // These are likely to be empty or have only the welcome message
      const potentialEmptyChats = chats.filter(
        (chat) => chat.title === "New Conversation" && chat.id !== activeChat
      );

      // Check each potential empty chat to confirm it's actually empty
      if (potentialEmptyChats.length > 0) {
        for (const chat of potentialEmptyChats) {
          try {
            // Fetch messages for this potential empty chat
            const response = await fetch(`/api/chats/${chat.id}/messages`);
            if (response.ok) {
              const chatMessages = await response.json();

              // Check if it only has the welcome message
              if (
                chatMessages.length === 1 &&
                chatMessages[0].role === "assistant" &&
                chatMessages[0].id === "welcome"
              ) {
                // Found an empty chat with just the welcome message
                emptyChat = chat;
                break; // Stop searching once we find one
              }
            }
          } catch (error) {
            console.error("Error checking chat messages:", error);
          }
        }
      }

      if (emptyChat) {
        // Use the existing empty chat instead of creating a new one
        // This is more efficient than creating a new chat
        console.log("Using existing empty chat:", emptyChat.id);

        // Set this empty chat as the active chat
        setActiveChat(emptyChat.id);

        // Fetch and set the messages for this chat (should just be the welcome message)
        const response = await fetch(`/api/chats/${emptyChat.id}/messages`);
        if (response.ok) {
          const chatMessages = await response.json();
          setMessages(chatMessages);
        } else {
          // If we can't fetch messages, just set an empty array
          setMessages([]);
        }
      } else {
        // Create a new chat if no empty chat exists
        await createNewChat();
      }
    } catch (error) {
      // Handle any errors that occur during the process
      console.error("Error creating new chat:", error);
      toast.error(
        "Cannot connect to the local server. The application may need to be restarted."
      );
    } finally {
      // Hide loading indicator regardless of success/failure
      setIsLoading(false);
    }
  };

  /**
   * Handles model selection
   *
   * This function:
   * 1. Updates the current model type state
   * 2. Shows a toast notification to indicate the model has been selected
   *
   * @param {string} modelType - The type of model selected
   * @param {string} defaultPrompt - The default prompt for the selected model
   */
  const handleModelSelect = (modelType: string, defaultPrompt: string) => {
    // Update the current model type
    setCurrentModelType(modelType);

    // Log the model selection for debugging
    console.log(`Selected model: ${modelType}`);
  };

  /**
   * Handles chat deletion
   *
   * This function:
   * 1. Removes the specified chat from the chats list
   * 2. If the deleted chat was active, selects another chat to display
   * 3. Clears messages if no chats remain
   *
   * Note: This function only updates the UI state. The actual deletion
   * is handled by the parent component that calls this function.
   *
   * @param {string} chatId - The ID of the chat to delete
   */
  const handleDeleteChat = (chatId: string) => {
    // Remove the chat from the list of chats
    setChats((prevChats) => prevChats.filter((chat) => chat.id !== chatId));

    // If the deleted chat was the active chat, we need to select another one
    if (chatId === activeChat) {
      // Get the list of remaining chats after deletion
      const remainingChats = chats.filter((chat) => chat.id !== chatId);

      if (remainingChats.length > 0) {
        // If there are other chats, select the first one
        setActiveChat(remainingChats[0].id);
      } else {
        // If no chats remain, clear the active chat and messages
        setActiveChat("");
        setMessages([]);
      }
    }
  };

  /**
   * Render the chat interface
   *
   * The UI is structured as follows:
   * 1. A sidebar for chat navigation and management
   * 2. A main content area with:
   *    - Header with status indicators and settings
   *    - Message display area
   *    - Input area for sending messages
   *
   * The layout is responsive and adapts to different screen sizes.
   */
  return (
    <div className="h-screen flex overflow-hidden relative">
      {/*
        Sidebar hover indicator - simplified without hover effects
        This creates a small invisible area on the left edge that can be used
        to trigger the sidebar on mobile devices
      */}
      <div
        className="fixed top-0 left-0 w-4 h-full z-10"
        style={{ pointerEvents: "auto" }}
      />

      {/*
        Chat Sidebar Component
        Displays the list of chats and provides controls for chat management
      */}
      <ChatSidebar
        chats={chats} // List of all chats
        activeChat={activeChat} // Currently selected chat
        onSelectChat={setActiveChat} // Handler for selecting a chat
        onNewChat={handleNewChat} // Handler for creating a new chat
        onDeleteChat={handleDeleteChat} // Handler for deleting a chat
      />

      {/*
        Main Content Area
        Contains the header, message display, and input area
      */}
      <main
        className="flex-1 flex flex-col h-full"
        style={{ width: windowSize.width ? `${windowSize.width}px` : "100%" }}
      >
        {/*
          Header Bar
          Contains status indicators and settings controls
        */}
        <header className="h-16 border-b flex items-center justify-between px-6 glass">
          {/* Left side of header - Status indicators */}
          <div className="flex-1">
            {/* Offline indicator - only shown when offline */}
            {isOffline && (
              <div className="flex items-center text-destructive">
                <span className="inline-block w-2 h-2 rounded-full bg-destructive mr-2 animate-pulse"></span>
                <span className="text-sm font-medium">Offline</span>
              </div>
            )}
          </div>

          {/* Right side of header - Theme toggle and settings panel */}
          <div className="flex items-center space-x-2">
            <ThemeToggle /> {/* Toggle between light/dark themes */}
            <SettingsPanel /> {/* Access application settings */}
          </div>
        </header>

        {/*
          Message Display Area
          Shows all messages in the current chat with scrolling
        */}
        <div className="flex-1 overflow-auto p-6 scrollbar-themed">
          {/* Container with responsive width based on screen size */}
          <div
            className={`mx-auto ${isMobile ? "w-full" : "max-w-3xl"}`}
            style={{
              maxWidth:
                windowSize.width && windowSize.width < 768
                  ? "100%" // Full width on mobile
                  : windowSize.width && windowSize.width < 1200
                  ? "85%" // 85% width on tablets
                  : "3xl", // Max width on desktops
            }}
          >
            {/* Empty state when no messages are present */}
            {messages.length === 0 && !isLoading ? (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                <p className="text-lg">No messages yet</p>
                <p className="text-sm">Start a conversation with Jarvis</p>
              </div>
            ) : (
              /* Map through messages and render each one */
              messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  role={msg.role} // 'user' or 'assistant'
                  content={msg.content} // Message text content
                  timestamp={msg.timestamp} // Time the message was sent
                />
              ))
            )}

            {/* Loading indicator shown when waiting for a response */}
            {isLoading && (
              <div className="flex justify-center my-4 animate-fade-in">
                <div className="flex space-x-3 items-center glass px-4 py-2 rounded-full shadow-sm">
                  <div className="text-sm text-muted-foreground">
                    Jarvis is thinking
                  </div>
                  <div className="flex space-x-1">
                    <div className="h-2 w-2 bg-primary rounded-full animate-[pulse_0.8s_ease-in-out_infinite]"></div>
                    <div className="h-2 w-2 bg-primary rounded-full animate-[pulse_0.8s_ease-in-out_0.2s_infinite]"></div>
                    <div className="h-2 w-2 bg-primary rounded-full animate-[pulse_0.8s_ease-in-out_0.4s_infinite]"></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/*
          Input Area
          Contains the chat input component for sending messages
        */}
        <div className="p-6 border-t">
          {/* Container with responsive width based on screen size */}
          <div
            className={`mx-auto ${isMobile ? "w-full" : "max-w-3xl"}`}
            style={{
              maxWidth:
                windowSize.width && windowSize.width < 768
                  ? "100%" // Full width on mobile
                  : windowSize.width && windowSize.width < 1200
                  ? "85%" // 85% width on tablets
                  : "3xl", // Max width on desktops
            }}
          >
            {/* Chat input component */}
            <ChatInput
              onSend={handleSendMessage} // Handler for sending messages
              isLoading={isLoading} // Whether a message is being processed
              disabled={isOffline} // Disable input when offline
              placeholder={
                isOffline
                  ? "You are offline. Cannot send messages."
                  : "Ask Jarvis anything..."
              }
              showModelButtons={true} // Show model selection buttons
              onModelSelect={handleModelSelect} // Handler for model selection
            />
          </div>
        </div>
      </main>
    </div>
  );
}
