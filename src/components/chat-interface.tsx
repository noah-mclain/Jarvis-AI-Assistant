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

type Chat = {
  id: string;
  title: string;
  date: string;
};

export function ChatInterface() {
  const [activeChat, setActiveChat] = useState<string>("");
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isOffline, setIsOffline] = useState(false);
  const windowSize = useWindowSize();
  const isMobile = useIsMobile();

  // Since we're running fully offline, we don't need to check online status
  // This is a local application that doesn't require internet connectivity

  // Initialize the application in "online" mode since we're connecting to a local server
  useEffect(() => {
    // Always set to false since we're running locally
    setIsOffline(false);
  }, []);

  // Fetch chats on component mount and handle chat initialization
  useEffect(() => {
    const fetchChats = async () => {
      try {
        const response = await fetch("/api/chats");
        if (response.ok) {
          const data = await response.json();
          setChats(data);

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
                  setActiveChat(mostRecentChat.id);
                  setMessages(messagesData);
                  console.log("Using existing empty chat:", mostRecentChat.id);
                } else if (
                  messagesData.length > 1 ||
                  (messagesData.length === 1 && messagesData[0].role === "user")
                ) {
                  // If the most recent chat has user messages, create a new one
                  await createNewChat();
                  console.log(
                    "Most recent chat has user messages, creating new chat"
                  );
                } else {
                  // Use the existing chat
                  setActiveChat(mostRecentChat.id);
                  setMessages(messagesData);
                  console.log("Using existing chat:", mostRecentChat.id);
                }
              } else {
                // If we can't fetch messages, create a new chat to be safe
                await createNewChat();
              }
            } else {
              // No existing chats, create a new one
              await createNewChat();
              console.log("No existing chats, creating new chat");
            }
          }
        } else {
          const errorData = await response.json().catch(() => ({}));
          console.error(
            "Failed to fetch chats:",
            errorData.error || response.statusText
          );
          toast.error("Failed to load chats. Please try again.");
        }
      } catch (error) {
        console.error("Error fetching chats:", error);
        toast.error(
          "Cannot connect to the local server. The application may need to be restarted."
        );
      } finally {
        setIsInitialized(true);
      }
    };

    fetchChats();
  }, []); // No dependencies to prevent re-fetching

  // Function to create a new chat
  const createNewChat = async () => {
    if (isOffline) return;

    try {
      const response = await fetch("/api/chats", {
        method: "POST",
      });

      if (response.ok) {
        const newChat = await response.json();
        setChats((prev) => [...prev, newChat]);
        setActiveChat(newChat.id);
        setMessages([]);
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error(
          "Failed to create new chat:",
          errorData.error || response.statusText
        );
      }
    } catch (error) {
      console.error("Error creating new chat:", error);
    }
  };

  // Fetch messages when active chat changes
  useEffect(() => {
    if (activeChat && isInitialized) {
      const fetchMessages = async () => {
        setIsLoading(true);
        try {
          const response = await fetch(`/api/chats/${activeChat}/messages`);
          if (response.ok) {
            const data = await response.json();
            setMessages(data);
          } else {
            const errorData = await response.json().catch(() => ({}));
            console.error(
              "Failed to fetch messages:",
              errorData.error || response.statusText
            );
            toast.error("Failed to load messages. Please try again.");
          }
        } catch (error) {
          console.error("Error fetching messages:", error);
          toast.error(
            "Cannot connect to the local server. The application may need to be restarted."
          );
        } finally {
          setIsLoading(false);
        }
      };

      fetchMessages();
    }
  }, [activeChat, isInitialized]);

  const handleSendMessage = async (message: string) => {
    if (!activeChat) return;

    if (isOffline) {
      toast.error("You are offline. Cannot send messages.");
      return;
    }

    // Add user message to UI immediately for better UX
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: message,
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send message to backend
      const response = await fetch(`/api/chats/${activeChat}/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ content: message }),
      });

      if (response.ok) {
        const data = await response.json();

        // Update chat title if needed
        if (data.chatTitleUpdated) {
          setChats((prevChats) =>
            prevChats.map((chat) =>
              chat.id === activeChat ? { ...chat, title: data.newTitle } : chat
            )
          );
        }

        // Add AI response from backend
        setMessages((prev) => [
          ...prev.filter((msg) => msg.id !== userMessage.id),
          data.userMessage,
          data.aiResponse,
        ]);
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error(
          "Failed to send message:",
          errorData.error || response.statusText
        );
        toast.error("Failed to send message. Please try again.");

        // Remove the temporary user message
        setMessages((prev) => prev.filter((msg) => msg.id !== userMessage.id));
      }
    } catch (error) {
      console.error("Error sending message:", error);
      toast.error(
        "Cannot connect to the local server. The application may need to be restarted."
      );

      // Remove the temporary user message
      setMessages((prev) => prev.filter((msg) => msg.id !== userMessage.id));
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = async () => {
    if (isOffline) {
      toast.error("You are offline. Cannot create new chat.");
      return;
    }

    setIsLoading(true);

    try {
      // Check if current chat is empty or has only the welcome message
      const hasOnlyWelcomeMessage =
        messages.length === 0 ||
        (messages.length === 1 &&
          messages[0].role === "assistant" &&
          messages[0].id === "welcome");

      if (activeChat && hasOnlyWelcomeMessage) {
        // Current chat is already empty or has only the welcome message, just keep using it
        console.log(
          "Current chat is already empty or has only the welcome message, no need to create a new one"
        );
        setIsLoading(false);
        return;
      }

      // Check if there's already an empty chat in the list
      let emptyChat = null;

      // Filter potential empty chats (with title "New Conversation")
      const potentialEmptyChats = chats.filter(
        (chat) => chat.title === "New Conversation" && chat.id !== activeChat
      );

      // Check each potential empty chat
      if (potentialEmptyChats.length > 0) {
        for (const chat of potentialEmptyChats) {
          try {
            const response = await fetch(`/api/chats/${chat.id}/messages`);
            if (response.ok) {
              const chatMessages = await response.json();
              if (
                chatMessages.length === 1 &&
                chatMessages[0].role === "assistant" &&
                chatMessages[0].id === "welcome"
              ) {
                // Found an empty chat with just the welcome message
                emptyChat = chat;
                break;
              }
            }
          } catch (error) {
            console.error("Error checking chat messages:", error);
          }
        }
      }

      if (emptyChat) {
        // Use the existing empty chat instead of creating a new one
        console.log("Using existing empty chat:", emptyChat.id);
        setActiveChat(emptyChat.id);
        const response = await fetch(`/api/chats/${emptyChat.id}/messages`);
        if (response.ok) {
          const chatMessages = await response.json();
          setMessages(chatMessages);
        } else {
          setMessages([]);
        }
      } else {
        // Create a new chat if no empty chat exists
        await createNewChat();
      }
    } catch (error) {
      console.error("Error creating new chat:", error);
      toast.error(
        "Cannot connect to the local server. The application may need to be restarted."
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Handle chat deletion
  const handleDeleteChat = (chatId: string) => {
    // Remove the chat from the list
    setChats((prevChats) => prevChats.filter((chat) => chat.id !== chatId));

    // If the deleted chat was active, select another chat
    if (chatId === activeChat) {
      const remainingChats = chats.filter((chat) => chat.id !== chatId);
      if (remainingChats.length > 0) {
        setActiveChat(remainingChats[0].id);
      } else {
        setActiveChat("");
        setMessages([]);
      }
    }
  };

  return (
    <div className="h-screen flex overflow-hidden relative">
      {/* Sidebar hover indicator - simplified without hover effects */}
      <div
        className="fixed top-0 left-0 w-4 h-full z-10"
        style={{ pointerEvents: "auto" }}
      />

      <ChatSidebar
        chats={chats}
        activeChat={activeChat}
        onSelectChat={setActiveChat}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
      />

      <main
        className="flex-1 flex flex-col h-full"
        style={{ width: windowSize.width ? `${windowSize.width}px` : "100%" }}
      >
        <header className="h-16 border-b flex items-center justify-between px-6 glass">
          {/* Left side of header */}
          <div className="flex-1">
            {isOffline && (
              <div className="flex items-center text-destructive">
                <span className="inline-block w-2 h-2 rounded-full bg-destructive mr-2 animate-pulse"></span>
                <span className="text-sm font-medium">Offline</span>
              </div>
            )}
          </div>

          {/* Right side of header - settings and theme icons */}
          <div className="flex items-center space-x-2">
            <ThemeToggle />
            <SettingsPanel />
          </div>
        </header>

        <div className="flex-1 overflow-auto p-6 scrollbar-themed">
          <div
            className={`mx-auto ${isMobile ? "w-full" : "max-w-3xl"}`}
            style={{
              maxWidth:
                windowSize.width && windowSize.width < 768
                  ? "100%"
                  : windowSize.width && windowSize.width < 1200
                  ? "85%"
                  : "3xl",
            }}
          >
            {messages.length === 0 && !isLoading ? (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                <p className="text-lg">No messages yet</p>
                <p className="text-sm">Start a conversation with Jarvis</p>
              </div>
            ) : (
              messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  role={msg.role}
                  content={msg.content}
                  timestamp={msg.timestamp}
                />
              ))
            )}

            {isLoading && (
              <div className="flex justify-center my-4">
                <div className="animate-pulse flex space-x-2">
                  <div className="h-2 w-2 bg-primary rounded-full"></div>
                  <div className="h-2 w-2 bg-primary rounded-full"></div>
                  <div className="h-2 w-2 bg-primary rounded-full"></div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="p-6 border-t">
          <div
            className={`mx-auto ${isMobile ? "w-full" : "max-w-3xl"}`}
            style={{
              maxWidth:
                windowSize.width && windowSize.width < 768
                  ? "100%"
                  : windowSize.width && windowSize.width < 1200
                  ? "85%"
                  : "3xl",
            }}
          >
            <ChatInput
              onSend={handleSendMessage}
              isLoading={isLoading}
              disabled={isOffline}
              placeholder={
                isOffline
                  ? "You are offline. Cannot send messages."
                  : "Ask Jarvis anything..."
              }
            />
          </div>
        </div>
      </main>
    </div>
  );
}
