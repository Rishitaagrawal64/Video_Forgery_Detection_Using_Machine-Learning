import { useEffect, useState } from "react";
import UploadDropzone from "./components/UploadDropzone";
import ResultCard from "./components/ResultCard";
import { uploadVideo, analyzeVideo, getJob, getModelInfo } from "./api/client";
import type { Job, ModelInfo } from "./types";

export default function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState<ModelInfo | null>(null);

  useEffect(() => {
    // Optional extra: the app still works if the endpoint is unavailable.
    getModelInfo().then(setModel).catch(() => setModel(null));
  }, []);

  async function handleFile(file: File) {
    setMsg("");
    setJob(null);
    setBusy(true);
    try {
      const { video_id } = await uploadVideo(file);
      let current = await analyzeVideo(video_id);
      setJob(current);
      while (current.status !== "completed" && current.status !== "failed") {
        await new Promise((r) => setTimeout(r, 1500));
        current = await getJob(video_id);
        setJob(current);
      }
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  const res = job?.result;

  return (
    <main className="mx-auto max-w-xl p-4 sm:p-8">
      <h1 className="text-2xl font-bold">Video Forgery Detector</h1>
      <p className="mt-1 text-sm text-gray-600">
        Upload a video (up to 50 MB) to check whether it is AI-generated.
      </p>

      <div className="mt-6">
        <UploadDropzone onFile={handleFile} disabled={busy} />
      </div>

      {msg && <p className="mt-4 rounded bg-red-50 p-3 text-red-700">{msg}</p>}
      {busy && !res && (
        <p className="mt-4 text-gray-700">Status: {job?.status ?? "uploading"}...</p>
      )}
      {job?.status === "failed" && (
        <p className="mt-4 rounded bg-red-50 p-3 text-red-700">
          {job.error ?? "Analysis failed"}
        </p>
      )}

      {res && (
        <div className="mt-6">
          <ResultCard result={res} threshold={model?.decision_threshold} />
        </div>
      )}

      {model && (
        <p className="mt-8 text-xs text-gray-500">
          Model: {model.predictor} · {model.aggregation} of {model.num_frames} frames · decision threshold{" "}
          {model.decision_threshold.toFixed(2)}
        </p>
      )}
    </main>
  );
}