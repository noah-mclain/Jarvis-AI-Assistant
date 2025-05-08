import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/components/theme-provider";
import { FontSizeProvider } from "@/components/font-size-provider";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";

// Configure QueryClient for offline-first operation
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Don't retry failed queries - we're offline
      retry: false,
      // Cache data for a long time since we're offline
      staleTime: Infinity,
      // Don't refetch on window focus since we're offline
      refetchOnWindowFocus: false,
      // Don't refetch on reconnect since we're offline
      refetchOnReconnect: false,
    },
  },
});

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <FontSizeProvider>
        <TooltipProvider>
          <Toaster />
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </FontSizeProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
