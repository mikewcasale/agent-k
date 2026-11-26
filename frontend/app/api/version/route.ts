import { connection } from "next/server";

export async function GET() {
  await connection();
  const revision = process.env.RENDER_GIT_COMMIT?.trim() || null;

  return Response.json(
    { revision },
    {
      headers: {
        "Cache-Control": "no-store, max-age=0",
      },
    }
  );
}
