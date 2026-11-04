import type { Job } from "../types";

interface Props {
  jobs: Job[];
  onDelete: (id: string) => void;
}

export default function HistoryList({ jobs, onDelete }: Props) {
  if (jobs.length === 0) {
    return <p className="text-sm text-slate-500">No analyses yet.</p>;
  }
  return (
    <ul className="divide-y divide-slate-200 rounded-xl border border-slate-200">
      {jobs.map((j) => (
        <li key={j.video_id} className="flex items-center justify-between gap-3 p-3">
          <div className="min-w-0">
            <p className="truncate text-slate-800">{j.filename}</p>
            <p className="text-xs text-slate-500">
              {new Date(j.created_at).toLocaleString()} · {j.status}
              {j.result && ` · ${Math.round(j.result.forged_probability * 100)}% forged`}
            </p>
          </div>
          <button
            onClick={() => onDelete(j.video_id)}
            className="text-sm text-red-700 hover:underline"
            aria-label={`Delete ${j.filename}`}
          >
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}