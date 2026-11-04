import type { JobStatus } from "../types";

const TEXT: Record<JobStatus, string> = {
  uploaded: "Uploaded. Waiting to start…",
  processing: "Analysing frames…",
  completed: "Analysis complete.",
  failed: "Analysis failed.",
};

export default function ProgressStatus({ status }: { status: JobStatus }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-200 p-4" role="status">
      {status === "processing" && (
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent" />
      )}
      <span className="text-slate-700">{TEXT[status]}</span>
    </div>
  );
}