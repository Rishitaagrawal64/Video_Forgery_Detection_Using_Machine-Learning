export type JobStatus = "uploaded" | "processing" | "completed" | "failed";

export interface AnalysisResult {
  label: "REAL" | "AI_GENERATED_FORGED";
  forged_probability: number;
  frame_scores: number[];
  frames_analyzed: number;
  video: { width: number; height: number; fps: number; frame_count: number; duration_sec: number };
  model: string;
  processing_sec: number | null;
}

export interface Job {
  video_id: string;
  filename: string;
  status: JobStatus;
  created_at: string;
  result: AnalysisResult | null;
  error: string | null;
}