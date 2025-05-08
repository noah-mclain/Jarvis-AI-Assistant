/**
 * Chat Sidebar Component
 *
 * This component provides a sidebar for navigating between different chat conversations.
 * It includes features like:
 * - Collapsible sidebar with toggle button
 * - List of available chats sorted by date
 * - New chat button
 * - Delete functionality with confirmation dialog
 * - Responsive design for mobile and desktop
 *
 * The sidebar is designed to work offline and integrates with the main chat interface.
 *
 * @module ChatSidebar
 * @author Nada Mohamed
 */

import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Trash2 } from "lucide-react";
import { toast } from "@/components/ui/sonner";
import { useWindowSize } from "@/hooks/use-window-size";
import { useIsMobile } from "@/hooks/use-mobile";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

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
 * Props for the ChatSidebar component
 *
 * @interface ChatSidebarProps
 * @property {Chat[]} chats - Array of chat objects to display in the sidebar
 * @property {string} [activeChat] - ID of the currently active chat
 * @property {function} onSelectChat - Callback function when a chat is selected
 * @property {function} onNewChat - Callback function when the new chat button is clicked
 * @property {function} [onDeleteChat] - Optional callback function when a chat is deleted
 */
type ChatSidebarProps = {
  chats: Chat[];
  activeChat?: string;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat?: (id: string) => void;
};

/**
 * ChatSidebar Component
 *
 * A sidebar component that displays a list of chat conversations and provides
 * controls for creating new chats and managing existing ones.
 *
 * @param {ChatSidebarProps} props - Component props
 * @returns {JSX.Element} The rendered chat sidebar component
 */
export function ChatSidebar({
  chats,
  activeChat,
  onSelectChat,
  onNewChat,
  onDeleteChat,
}: ChatSidebarProps) {
  // State for sidebar open/closed status - starts closed by default
  const [isOpen, setIsOpen] = useState(false);

  // State for tracking which chat is being deleted
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);

  // State for controlling the delete confirmation dialog
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

  // Custom hooks for responsive design
  const windowSize = useWindowSize(); // Tracks window dimensions
  const isMobile = useIsMobile(); // Determines if viewport is mobile size

  /**
   * Toggles the sidebar open/closed state
   *
   * This function is called when the user clicks the sidebar toggle button.
   * It simply inverts the current open state of the sidebar.
   */
  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  /**
   * Handles the click on a chat's delete button
   *
   * This function:
   * 1. Stops event propagation to prevent selecting the chat
   * 2. Sets the chat ID to be deleted in state
   * 3. Opens the delete confirmation dialog
   *
   * @param {string} chatId - The ID of the chat to delete
   * @param {React.MouseEvent} e - The click event
   */
  const handleDeleteClick = (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering the chat selection
    setChatToDelete(chatId);
    setIsDeleteDialogOpen(true);
  };

  /**
   * Confirms and executes chat deletion
   *
   * This function:
   * 1. Validates that we have a chat to delete and a delete handler
   * 2. Makes an API call to delete the chat from the server
   * 3. Calls the parent component's delete handler on success
   * 4. Handles errors and shows appropriate notifications
   * 5. Cleans up the dialog state regardless of outcome
   *
   * @returns {Promise<void>}
   */
  const confirmDelete = async () => {
    // Validate that we have a chat to delete and a delete handler
    if (!chatToDelete || !onDeleteChat) {
      return;
    }

    try {
      // Call the API to delete the chat from the server
      const response = await fetch(`/api/chats/${chatToDelete}`, {
        method: "DELETE",
      });

      if (response.ok) {
        // Call the parent component's delete handler to update UI
        onDeleteChat(chatToDelete);
        toast.success("Chat deleted successfully");
      } else {
        // Handle API error
        const errorData = await response.json().catch(() => ({}));
        console.error(
          "Failed to delete chat:",
          errorData.error || response.statusText
        );
        toast.error("Failed to delete chat. Please try again.");
      }
    } catch (error) {
      // Handle network or server errors
      console.error("Error deleting chat:", error);
      toast.error("Failed to delete chat. Please try again.");
    } finally {
      // Clean up state regardless of outcome
      setIsDeleteDialogOpen(false);
      setChatToDelete(null);
    }
  };

  /**
   * Render the chat sidebar
   *
   * The sidebar UI consists of:
   * 1. A container with animation for showing/hiding
   * 2. A header with the application title
   * 3. A scrollable list of chats
   * 4. A new chat button
   * 5. A delete confirmation dialog
   * 6. A toggle button that changes position based on sidebar state
   *
   * The sidebar is responsive and adapts to different screen sizes.
   */
  return (
    <div className="relative h-screen">
      {/* Main sidebar container with animation for showing/hiding */}
      <div
        className={cn(
          "fixed top-0 left-0 h-full bg-sidebar border-r border-sidebar-border z-20",
          "transition-all duration-400 will-change-transform",
          "motion-reduce:transition-none" /* Respect reduced motion preferences */,
          // Width changes based on open state and device size
          isOpen ? (isMobile ? "w-full sm:w-72" : "w-64") : "w-0"
        )}
        style={{
          // Hardware acceleration for smoother animations
          transform: isOpen
            ? "translateX(0) translateZ(0)"
            : "translateX(-10px) translateZ(0)",
          backfaceVisibility: "hidden",
          // Smooth transition properties
          transition:
            "transform 400ms cubic-bezier(0.16, 1, 0.3, 1), width 400ms cubic-bezier(0.16, 1, 0.3, 1), box-shadow 400ms cubic-bezier(0.16, 1, 0.3, 1)",
          // Add subtle shadow when open
          boxShadow: isOpen ? "0 4px 20px rgba(0, 0, 0, 0.1)" : "none",
        }}
      >
        {/* Inner container with fade animation */}
        <div
          className={cn(
            "h-full flex flex-col",
            "will-change-opacity will-change-transform",
            "motion-reduce:transition-none" /* Respect reduced motion preferences */,
            // Fade in/out based on open state
            isOpen ? "opacity-100" : "opacity-0"
          )}
          style={{
            // Hardware acceleration for smoother animations
            transform: isOpen
              ? "translateX(0) translateZ(0)"
              : "translateX(-20px) translateZ(0)",
            backfaceVisibility: "hidden",
            // Stagger the content animation slightly after the sidebar opens
            transition:
              "opacity 350ms cubic-bezier(0.16, 1, 0.3, 1) 50ms, transform 350ms cubic-bezier(0.16, 1, 0.3, 1) 50ms",
            // Prevent content from showing during closed state
            pointerEvents: isOpen ? "auto" : "none",
          }}
        >
          {/* Sidebar header with application title */}
          <div
            className="p-4 border-b border-sidebar-border flex items-center"
            style={{
              // Add animation for the header
              opacity: isOpen ? 1 : 0,
              transform: isOpen ? "translateY(0)" : "translateY(-5px)",
              transition:
                "opacity 300ms cubic-bezier(0.16, 1, 0.3, 1) 50ms, transform 300ms cubic-bezier(0.16, 1, 0.3, 1) 50ms",
            }}
          >
            <h2 className="font-semibold gradient-text chat-sidebar-title">
              Jarvis AI
            </h2>
          </div>

          {/* Scrollable content area */}
          <div className="flex-1 overflow-auto py-2 scrollbar-themed">
            {/* New chat button */}
            <div className="px-3 mb-3">
              <button
                onClick={onNewChat}
                className="w-full py-2 px-3 rounded-lg bg-primary/10 hover:bg-primary/20 hover:shadow-md transition-all duration-200 text-primary font-medium"
                style={{
                  // Add animation for the new chat button
                  opacity: isOpen ? 1 : 0,
                  transform: isOpen ? "translateY(0)" : "translateY(-10px)",
                  transition:
                    "opacity 300ms cubic-bezier(0.16, 1, 0.3, 1) 100ms, transform 300ms cubic-bezier(0.16, 1, 0.3, 1) 100ms, background-color 200ms ease, box-shadow 200ms ease",
                }}
              >
                + New Chat
              </button>
            </div>

            {/* List of chats */}
            <div className="space-y-1 px-3">
              {/* Sort chats by date (newest first) and map to UI elements */}
              {[...chats]
                .sort((a, b) => {
                  // First try to parse the date
                  const dateA = new Date(a.date);
                  const dateB = new Date(b.date);

                  // If dates are the same, compare by ID (which contains timestamp)
                  if (dateA.getTime() === dateB.getTime()) {
                    // Extract timestamp from ID if possible
                    const timeA = a.id.includes("-")
                      ? parseInt(a.id.split("-").pop() || "0")
                      : 0;
                    const timeB = b.id.includes("-")
                      ? parseInt(b.id.split("-").pop() || "0")
                      : 0;
                    return timeB - timeA; // Descending order (newest first)
                  }

                  // Sort by date in descending order (newest first)
                  return dateB.getTime() - dateA.getTime();
                })
                .map((chat, index) => (
                  // Individual chat item with hover group for delete button
                  <div
                    key={chat.id}
                    className="relative group"
                    style={{
                      // Add staggered animation for each chat item
                      opacity: isOpen ? 1 : 0,
                      transform: isOpen ? "translateX(0)" : "translateX(-10px)",
                      transition: `opacity 300ms cubic-bezier(0.16, 1, 0.3, 1) ${
                        150 + index * 30
                      }ms, transform 300ms cubic-bezier(0.16, 1, 0.3, 1) ${
                        150 + index * 30
                      }ms`,
                    }}
                  >
                    {/* Chat item button */}
                    <button
                      onClick={() => onSelectChat(chat.id)}
                      className={cn(
                        "w-full py-2 px-3 text-left rounded-lg",
                        "transition-all duration-200",
                        // Highlight active chat
                        chat.id === activeChat
                          ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-sm"
                          : "text-sidebar-foreground hover:bg-sidebar-accent/50 hover:shadow-sm"
                      )}
                    >
                      {/* Chat title */}
                      <div className="font-medium truncate pr-6 chat-sidebar-item">
                        {chat.title}
                      </div>
                      {/* Chat date */}
                      <div className="text-muted-foreground truncate chat-sidebar-date">
                        {chat.date}
                      </div>
                    </button>

                    {/* Delete button - only shown if onDeleteChat is provided */}
                    {onDeleteChat && (
                      <button
                        onClick={(e) => handleDeleteClick(chat.id, e)}
                        className="absolute right-2 top-2 p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive/10"
                        aria-label="Delete chat"
                      >
                        <Trash2 className="h-3.5 w-3.5 text-muted-foreground hover:text-destructive" />
                      </button>
                    )}
                  </div>
                ))}
            </div>

            {/* Delete Confirmation Dialog */}
            <AlertDialog
              open={isDeleteDialogOpen}
              onOpenChange={setIsDeleteDialogOpen}
            >
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will permanently delete this chat and all its messages.
                    This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={confirmDelete}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>
      </div>

      {/*
        Sidebar toggle button
        This button is always visible but changes position based on sidebar state
        and screen size. It also animates between hamburger and X icons.
      */}
      <button
        onClick={toggleSidebar}
        className={cn(
          "fixed z-30 flex items-center justify-center w-12 h-12 rounded-md",
          "will-change-transform",
          "motion-reduce:transition-none" /* Respect reduced motion preferences */,
          "active:scale-95 hover:bg-primary/10" /* Interactive feedback */,
          "sidebar-toggle",
          // Add class for X animation when sidebar is open
          isOpen ? "sidebar-open" : ""
        )}
        aria-label={isOpen ? "Close sidebar" : "Open sidebar"}
        style={{
          // Dynamic positioning based on sidebar state and screen size
          left: isOpen
            ? windowSize.width && windowSize.width < 640
              ? "calc(100% - 3rem)" // Mobile open position
              : "calc(16rem - 3rem)" // Desktop open position
            : windowSize.width && windowSize.width < 640
            ? "0.5rem" // Mobile closed position
            : "1rem", // Desktop closed position
          top: windowSize.width && windowSize.width < 640 ? "0.5rem" : "1rem",
          // Hardware acceleration for smoother animations
          transform: isOpen
            ? "translateZ(0) rotate(0deg)"
            : "translateZ(0) rotate(0deg)",
          backfaceVisibility: "hidden",
          // Smooth transition properties matching sidebar animation
          transition:
            "left 400ms cubic-bezier(0.16, 1, 0.3, 1), transform 400ms cubic-bezier(0.16, 1, 0.3, 1), background-color 200ms ease",
          // Background color with subtle animation
          backgroundColor: isOpen
            ? "rgba(var(--primary-rgb), 0.1)"
            : "rgba(var(--primary-rgb), 0.05)",
          // Add subtle shadow
          boxShadow: "0 2px 8px rgba(0, 0, 0, 0.05)",
        }}
      >
        {/* Hamburger icon with animation to X */}
        <div className="relative w-6 h-6 flex items-center justify-center">
          <span className="hamburger-top bg-foreground"></span>
          <span className="hamburger-middle bg-foreground"></span>
          <span className="hamburger-bottom bg-foreground"></span>
        </div>
      </button>
    </div>
  );
}
