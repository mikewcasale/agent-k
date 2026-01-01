"use client";

import { MissionDashboard } from "@/components/agent-k/mission-dashboard";
import { MissionStream } from "@/components/agent-k/mission-stream";
import type { ChatMessage } from "@/lib/types";
import type { AppUsage } from "@/lib/usage";
import type { VisibilityType } from "./visibility-selector";

export function Chat(_props: {
  id: string;
  initialMessages: ChatMessage[];
  initialChatModel: string;
  initialVisibilityType: VisibilityType;
  isReadonly: boolean;
  autoResume: boolean;
  initialLastContext?: AppUsage;
}) {
  return (
    <div className="min-h-dvh bg-background">
      <MissionStream />
      <div className="mx-auto w-full max-w-5xl px-4 py-6">
        <MissionDashboard />
      </div>
    </div>
  );
}
