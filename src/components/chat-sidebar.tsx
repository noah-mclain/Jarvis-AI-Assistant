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

type Chat = {
  id: string;
  title: string;
  date: string;
};

type ChatSidebarProps = {
  chats: Chat[];
  activeChat?: string;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat?: (id: string) => void;
};

export function ChatSidebar({
  chats,
  activeChat,
  onSelectChat,
  onNewChat,
  onDeleteChat,
}: ChatSidebarProps) {
  const [isOpen, setIsOpen] = useState(false); // Start with sidebar closed
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const windowSize = useWindowSize();
  const isMobile = useIsMobile();

  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  const handleDeleteClick = (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering the chat selection
    setChatToDelete(chatId);
    setIsDeleteDialogOpen(true);
  };

  const confirmDelete = async () => {
    if (!chatToDelete || !onDeleteChat) return;

    try {
      // Call the API to delete the chat
      const response = await fetch(`/api/chats/${chatToDelete}`, {
        method: "DELETE",
      });

      if (response.ok) {
        // Call the parent component's delete handler
        onDeleteChat(chatToDelete);
        toast.success("Chat deleted successfully");
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error(
          "Failed to delete chat:",
          errorData.error || response.statusText
        );
        toast.error("Failed to delete chat. Please try again.");
      }
    } catch (error) {
      console.error("Error deleting chat:", error);
      toast.error("Failed to delete chat. Please try again.");
    } finally {
      setIsDeleteDialogOpen(false);
      setChatToDelete(null);
    }
  };

  return (
    <div className="relative h-screen">
      <div
        className={cn(
          "fixed top-0 left-0 h-full bg-sidebar transition-all duration-500 ease-in-out border-r border-sidebar-border z-20",
          isOpen ? (isMobile ? "w-full sm:w-72" : "w-64") : "w-0"
        )}
      >
        <div
          className={cn(
            "h-full flex flex-col transition-opacity duration-500 ease-in-out",
            isOpen ? "opacity-100" : "opacity-0"
          )}
        >
          <div className="p-4 border-b border-sidebar-border flex items-center">
            <h2 className="font-semibold gradient-text chat-sidebar-title">
              Jarvis AI
            </h2>
          </div>

          <div className="flex-1 overflow-auto py-2 scrollbar-themed">
            <div className="px-3 mb-3">
              <button
                onClick={onNewChat}
                className="w-full py-2 px-3 rounded-lg bg-primary/10 hover:bg-primary/20 transition-colors text-primary font-medium"
              >
                + New Chat
              </button>
            </div>

            <div className="space-y-1 px-3">
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
                    return timeB - timeA; // Descending order
                  }

                  // Sort by date in descending order
                  return dateB.getTime() - dateA.getTime();
                })
                .map((chat) => (
                  <div key={chat.id} className="relative group">
                    <button
                      onClick={() => onSelectChat(chat.id)}
                      className={cn(
                        "w-full py-2 px-3 text-left rounded-lg transition-colors",
                        chat.id === activeChat
                          ? "bg-sidebar-accent text-sidebar-accent-foreground"
                          : "text-sidebar-foreground hover:bg-sidebar-accent/50"
                      )}
                    >
                      <div className="font-medium truncate pr-6 chat-sidebar-item">
                        {chat.title}
                      </div>
                      <div className="text-muted-foreground truncate chat-sidebar-date">
                        {chat.date}
                      </div>
                    </button>

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

      {/* Sidebar toggle button - visible in both states but with different positions */}
      <button
        onClick={toggleSidebar}
        className={cn(
          "fixed z-30 flex items-center justify-center w-12 h-12 rounded-md bg-primary/5 transition-all duration-500 ease-in-out active:scale-95 sidebar-toggle",
          isOpen ? "sidebar-open" : ""
        )}
        aria-label={isOpen ? "Close sidebar" : "Open sidebar"}
        style={{
          left: isOpen
            ? windowSize.width && windowSize.width < 640
              ? "calc(100% - 3rem)"
              : "calc(16rem - 3rem)"
            : windowSize.width && windowSize.width < 640
            ? "0.5rem"
            : "1rem",
          top: windowSize.width && windowSize.width < 640 ? "0.5rem" : "1rem",
        }}
      >
        <div className="relative w-6 h-6 flex items-center justify-center">
          {/* Using the new hamburger animation classes */}
          <span className="hamburger-top bg-foreground"></span>
          <span className="hamburger-middle bg-foreground"></span>
          <span className="hamburger-bottom bg-foreground"></span>
        </div>
      </button>
    </div>
  );
}
