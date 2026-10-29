import type { AnalyzeOptions, HealthResponse, Job } from "../types";
// analyzing 
const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";
const ROOT = BASE.replace(/\/api\/v1$/, "");

async function check(res: Response) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  }
  return res;
}

export async function uploadVideo(file: File): Promise<{ video_id: string; filename: string }> {
  const form = new FormData();
  form.append("file", file);
  return (await check(await fetch(`${BASE}/videos`, { method: "POST", body: form }))).json();
}

export async function analyzeVideo(id: string, options?: AnalyzeOptions): Promise<Job> {
  return (
    await check(
      await fetch(`${BASE}/videos/${id}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(options ?? {}),
      }),
    )
  ).json();
}

export async function getJob(id: string): Promise<Job> {
  return (await check(await fetch(`${BASE}/jobs/${id}`))).json();
}

export async function listJobs(): Promise<Job[]> {
  return (await check(await fetch(`${BASE}/jobs?limit=20&offset=0`))).json();
}

export async function deleteVideo(id: string): Promise<void> {
  await check(await fetch(`${BASE}/videos/${id}`, { method: "DELETE" }));
}

export async function getHealth(): Promise<HealthResponse> {
  return (await check(await fetch(`${ROOT}/health`))).json();
}