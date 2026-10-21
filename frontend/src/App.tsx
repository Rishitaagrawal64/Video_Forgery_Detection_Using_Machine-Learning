import { useState } from "react";

const API = "http://localhost:8000/api/v1";
const MAX_MB = 50;
const ALLOWED = [".mp4", ".avi", ".mov", ".mkv", ".webm"];

type Job = {
  video_id: string;
  status: string;
  error?: string | null;
  result?: {
    label: string;
    forged_probability: number;
    frame_scores: number[];
    processing_sec?: number | null;
    model: string;
  } | null;
};

export default function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleFile(file: File) {
    setMsg("");
    setJob(null);
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED.includes(ext)) return setMsg("Please choose an MP4, AVI, MOV, MKV or WebM video.");
    if (file.size > MAX_MB * 1024 * 1024) return setMsg(`The file is larger than ${MAX_MB} MB.`);
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const up = await fetch(`${API}/videos`, { method: "POST", body: form });
      if (!up.ok) throw new Error((await up.json()).detail ?? "Upload failed");
      const { video_id } = await up.json();

      const an = await fetch(`${API}/videos/${video_id}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      if (!an.ok) throw new Error((await an.json()).detail ?? "Could not start the analysis");

      for (;;) {
        const j: Job = await (await fetch(`${API}/jobs/${video_id}`)).json();
        setJob(j);
        if (j.status === "completed" || j.status === "failed") break;
        await new Promise((r) => setTimeout(r, 1500));
      }
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  const res = job?.result;
  const forged = res?.label === "AI_GENERATED_FORGED";

  return (
    <main className="mx-auto max-w-xl p-4 sm:p-8">
      <h1 className="text-2xl font-bold">Video Forgery Detector</h1>
      <p className="mt-1 text-sm text-gray-600">Upload a video (up to {MAX_MB} MB) to check whether it is AI-generated.</p>

      <input
        type="file"
        accept="video/*"
        disabled={busy}
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        className="mt-6 block w-full rounded border p-3"
      />

      {msg && <p className="mt-4 rounded bg-red-50 p-3 text-red-700">{msg}</p>}
      {busy && !res && <p className="mt-4 text-gray-700">Status: {job?.status ?? "uploading"}...</p>}
      {job?.status === "failed" && <p className="mt-4 rounded bg-red-50 p-3 text-red-700">{job.error}</p>}

      {res && (
        <section className="mt-6 rounded border p-4">
          <p className={`text-xl font-bold ${forged ? "text-red-600" : "text-green-700"}`}>
            {forged ? "AI-GENERATED FORGED" : "REAL"}
          </p>
          <p>Forged probability: {(res.forged_probability * 100).toFixed(1)}%</p>
          <p className="text-sm text-gray-600">Model: {res.model} | Time: {res.processing_sec ?? "?"} s</p>
          <div className="mt-3 flex h-20 items-end gap-1">
            {res.frame_scores.map((s, i) => (
              <div key={i} title={`Frame ${i + 1}: ${s.toFixed(2)}`} className="flex-1 bg-gray-300" style={{ height: `${s * 100}%` }} />
            ))}
          </div>
          <p className="mt-1 text-xs text-gray-500">Each bar is one analysed frame.</p>
        </section>
      )}
    </main>
  );
}