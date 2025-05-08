import { useTheme } from "@/components/theme-provider";
import { Toaster as Sonner, toast } from "sonner";

type ToasterProps = React.ComponentProps<typeof Sonner>;

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme } = useTheme();

  // Map our theme values to what sonner expects
  const sonnerTheme =
    theme === "system" ? "system" : theme === "dark" ? "dark" : "light";

  return (
    <Sonner
      theme={sonnerTheme as ToasterProps["theme"]}
      className="toaster group"
      closeButton={true} // Add close button
      duration={7000} // Auto-dismiss after 7 seconds
      position="bottom-right" // Position at the bottom right
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg relative group-[.toaster]:data-[state=open]:animate-in group-[.toaster]:data-[state=open]:fade-in-0 group-[.toaster]:data-[state=open]:slide-in-from-bottom-full group-[.toaster]:data-[state=closed]:animate-out group-[.toaster]:data-[state=closed]:fade-out-0 group-[.toaster]:data-[state=closed]:slide-out-to-bottom-full group-[.toaster]:data-[swipe=move]:translate-y-[var(--radix-toast-swipe-move-y)] group-[.toaster]:data-[swipe=cancel]:translate-y-0 group-[.toaster]:data-[swipe=end]:translate-y-[var(--radix-toast-swipe-end-y)]",
          description: "group-[.toast]:text-muted-foreground",
          actionButton:
            "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground",
          cancelButton:
            "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground",
          closeButton:
            "group-[.toast]:opacity-0 group-[.toast]:transition-opacity group-[.toast]:duration-300 group-[.toast]:absolute group-[.toast]:top-2 group-[.toast]:right-2 group-[.toast]:p-1 group-[.toast]:rounded-md group-[.toast]:text-foreground/50 group-[.toast]:hover:text-foreground group-[.toast]:hover:bg-accent group-[.toast]:focus:opacity-100 group-[.toast]:focus:outline-none group-[.toast]:focus:ring-2 group-[.toast]:group-hover:opacity-100",
        },
      }}
      {...props}
    />
  );
};

export { Toaster, toast };
