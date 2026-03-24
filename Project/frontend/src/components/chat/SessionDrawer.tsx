// components/chat/SessionDrawer.tsx
"use client";

import { useEffect, useRef } from "react";
import { SessionSummary } from "@/lib/api";

interface Props {
  open: boolean;
  onClose: () => void;
  sessions: SessionSummary[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  isCreating: boolean;
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(dateStr).toLocaleDateString([], { month: "short", day: "numeric" });
}

export default function SessionDrawer({
  open,
  onClose,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  isCreating,
}: Props) {
  const drawerRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (drawerRef.current && !drawerRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, onClose]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 z-40 bg-black/20 backdrop-blur-[2px] transition-opacity duration-300
          ${open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"}`}
        aria-hidden="true"
      />

      {/* Drawer panel */}
      <div
        ref={drawerRef}
        className={`fixed top-0 left-0 h-full z-50 w-72 flex flex-col
          transition-transform duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]
          ${open ? "translate-x-0" : "-translate-x-full"}`}
        style={{
          background: "rgba(255,255,255,0.92)",
          backdropFilter: "blur(20px)",
          borderRight: "1px solid rgba(139,92,246,0.12)",
          boxShadow: "4px 0 40px rgba(139,92,246,0.10)",
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-5 pb-3">
          <span
            className="text-sm font-semibold"
            style={{ background: "linear-gradient(to right, #8B5CF6, #D946EF)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}
          >
            Conversations
          </span>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors cursor-pointer"
            aria-label="Close drawer"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* New session button */}
        <div className="px-3 pb-3">
          <button
            onClick={onNewSession}
            disabled={isCreating}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl
              text-sm font-medium text-white transition-all duration-200
              hover:opacity-90 active:scale-95 disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer"
            style={{ background: "linear-gradient(to right, #8B5CF6, #D946EF)", boxShadow: "0 2px 12px rgba(139,92,246,0.35)" }}
          >
            {isCreating ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="white" strokeWidth="4" />
                <path className="opacity-75" fill="white" d="M4 12a8 8 0 018-8v8z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            )}
            New Session
          </button>
        </div>

        <div className="mx-3 mb-2 h-px bg-gradient-to-r from-transparent via-purple-200 to-transparent" />

        {/* Session list */}
        <div className="flex-1 overflow-y-auto px-2 pb-4 space-y-1">
          {sessions.length === 0 && (
            <p className="text-xs text-gray-400 text-center mt-8 px-4">
              No conversations yet. Start a new session!
            </p>
          )}
          {sessions.map((s) => {
            const isActive = s.session_id === activeSessionId;
            return (
              <div
                key={s.session_id}
                className={`group relative flex items-center gap-2 px-3 py-2.5 rounded-xl
                  cursor-pointer transition-all duration-150 select-none
                  ${isActive
                    ? "bg-purple-50 border border-purple-200/60"
                    : "hover:bg-gray-50 border border-transparent"
                  }`}
                onClick={() => { onSelectSession(s.session_id); onClose(); }}
              >
                {/* Active indicator */}
                {isActive && (
                  <span
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-full"
                    style={{ background: "linear-gradient(to bottom, #8B5CF6, #D946EF)" }}
                  />
                )}

                <div className="flex-1 min-w-0">
                  <p className={`text-sm truncate ${isActive ? "text-purple-700 font-medium" : "text-gray-700"}`}>
                    {s.first_message ?? "New conversation"}
                  </p>
                  <p className="text-[10px] text-gray-400 mt-0.5">{timeAgo(s.updated_at)}</p>
                </div>

                {/* Delete button — only on hover */}
                <button
                  onClick={(e) => { e.stopPropagation(); onDeleteSession(s.session_id); }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded-md text-gray-300
                    hover:text-red-400 hover:bg-red-50 transition-all duration-150 flex-shrink-0 cursor-pointer"
                  aria-label="Delete session"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}