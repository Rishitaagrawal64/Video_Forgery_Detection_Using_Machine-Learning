import { useEffect, useState } from "react";
import UploadDropzone from "./components/UploadDropzone";
import ResultCard from "./components/ResultCard";
import ProgressStatus from "./components/ProgressStatus";
import ErrorMessage from "./components/ErrorMessage";
import HistoryList from "./components/HistoryList";
import {
  uploadVideo,
  analyzeVideo,
  getJob,
  listJobs,
  deleteVideo,
  getModelInfo,
} from "./api/client";
import type { Job, ModelInfo } from "./types";

export default function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [history, setHistory] = useState<Job[]>([]);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState<ModelInfo | null>(null);

  useEffect(() => {
    // Optional extra: the app still works if the endpoint is unavailable.
    getModelInfo().then(setModel).catch(() => setModel(null));
  }, []);

  async function refreshHistory() {
    try {
      setHistory(await listJobs());
    } catch {
      /* history is optional; ignore errors */
    }
  }

  useEffect(() => {
    refreshHistory();
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
      refreshHistory();
    }
  }

  async function handleDelete(id: string) {
    try {
      await deleteVideo(id);
      if (job?.video_id === id) setJob(null);
      await refreshHistory();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Could not delete");
    }
  }

  const res = job?.result;
  const inProgress = job && job.status !== "completed" && job.status !== "failed";

  return (
    <main className="mx-auto max-w-xl p-4 sm:p-8">
      <h1 className="text-2xl font-bold">Video Forgery Detector</h1>
      <p className="mt-1 text-sm text-gray-600">
        Upload a video (up to 50 MB) to check whether it is AI-generated.
      </p>

      <div className="mt-6">
        <UploadDropzone onFile={handleFile} disabled={busy} />
      </div>

      {busy && !job && (
        <div className="mt-4">
          <ProgressStatus status="uploaded" />
        </div>
      )}
      {inProgress && (
        <div className="mt-4">
          <ProgressStatus status={job.status} />
        </div>
      )}
      {msg && (
        <div className="mt-4">
          <ErrorMessage message={msg} />
        </div>
      )}
      {job?.status === "failed" && (
        <div className="mt-4">
          <ErrorMessage message={job.error ?? "Analysis failed"} />
        </div>
      )}

      {res && (
        <div className="mt-6">
          <ResultCard result={res} threshold={model?.decision_threshold} />
        </div>
      )}

      <h2 className="mt-8 mb-2 text-lg font-semibold">History</h2>
      <HistoryList jobs={history} onDelete={handleDelete} />

      {model && (
        <p className="mt-8 text-xs text-gray-500">
          Model: {model.predictor} · {model.aggregation} of {model.num_frames} frames · decision threshold{" "}
          {model.decision_threshold.toFixed(2)}
        </p>
      )}
    </main>
  );
}