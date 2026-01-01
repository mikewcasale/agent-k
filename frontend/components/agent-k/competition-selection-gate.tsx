"use client";

import type React from "react";
import { Chat } from "@/components/chat";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import { CompetitionSelector } from "./competition-selector";

type CompetitionSelectionGateProps = {
  chatKey: string;
  chatProps: Omit<React.ComponentProps<typeof Chat>, "key">;
};

export function CompetitionSelectionGate({
  chatKey,
  chatProps,
}: CompetitionSelectionGateProps) {
  const { state } = useAgentKState();
  const hasMission =
    state.mission.phases.length > 0 ||
    state.mission.evolution ||
    state.mission.competition;

  if (!hasMission) {
    return <CompetitionSelector />;
  }

  return <Chat key={chatKey} {...chatProps} />;
}
