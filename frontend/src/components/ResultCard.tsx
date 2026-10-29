import type { AnalysisResult } from "../types";

export default function ResultCard({ result }: { result: AnalysisResult }) {
  const forged = result.label === "AI_GENERATED_FORGED";
  const pct = Math.min(100, Math.max(0, Math.round(result.forged_probability * 100)));
  const isPlaceholder = result.model.toLowerCase().includes("placeholder");

  return (
    <section
      className={`rounded-2xl p-6 ${
        forged ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
      }`}
      aria-label="Analysis result"
    >
      <p className="text-3xl font-bold">
        {forged ? "Likely AI-generated / forged" : "Likely real"}
      </p>
      <p className="mt-1">Forged probability: {pct}%</p>

      <div
        className="relative mt-4 h-3 rounded-full bg-white/70"
        role="meter"
        aria-label="Forged probability"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
      >
        <div className="h-3 rounded-full bg-current" style={{ width: `${pct}%` }} />
        <div
          className="absolute top-[-4px] h-5 w-0.5 bg-slate-700"
          style={{ left: "50%" }}
          title="50% decision threshold"
        />
      </div>

      <div className="mt-4 flex h-16 items-end gap-1" aria-label="Per-frame forged scores">
        {result.frame_scores.map((s, i) => (
          <div
            key={i}
            className="flex-1 rounded-sm bg-current opacity-60"
            style={{ height: `${Math.max(s * 100, 4)}%` }}
            title={`Frame ${i + 1}: ${s.toFixed(2)}`}
          />
        ))}
      </div>

      {isPlaceholder && (
        <p className="mt-4 rounded-lg bg-amber-100 p-3 text-sm text-amber-800" role="note">
          Demo mode: this result comes from a placeholder model, not the trained detector.
        </p>
      )}

      <p className="mt-4 text-sm text-slate-700">
        Model: {result.model} · {result.frames_analyzed} frames
        {result.processing_sec != null && ` · ${result.processing_sec.toFixed(1)} s`}.
        This is a statistical estimate, not proof.
      </p>
    </section>
  );
}