import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { Suspense } from "react";
import { CompetitionSelectionGate } from "@/components/agent-k/competition-selection-gate";
import { DataStreamHandler } from "@/components/data-stream-handler";
import { DEFAULT_CHAT_MODEL } from "@/lib/ai/models";
import { generateUUID } from "@/lib/utils";
import { auth } from "../(auth)/auth";

export default function Page() {
  return (
    <Suspense fallback={<div className="flex h-dvh" />}>
      <NewChatPage />
    </Suspense>
  );
}

async function NewChatPage() {
  const session = await auth();

  if (!session) {
    redirect("/api/auth/guest");
  }

  const id = generateUUID();

  const cookieStore = await cookies();
  const modelIdFromCookie = cookieStore.get("chat-model");

  const chatProps = {
    autoResume: false,
    id,
    initialChatModel: modelIdFromCookie?.value ?? DEFAULT_CHAT_MODEL,
    initialMessages: [],
    initialVisibilityType: "private" as const,
    isReadonly: false,
  };

  return (
    <>
      <CompetitionSelectionGate chatKey={id} chatProps={chatProps} />
      <DataStreamHandler />
    </>
  );
}
