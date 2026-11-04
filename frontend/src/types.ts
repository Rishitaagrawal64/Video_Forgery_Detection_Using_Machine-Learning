export type JobStatus = "uploaded" | "processing" | "completed" | "failed";
//
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
export interface AnalyzeOptions {
  num_frames?: number;   // 1-128; omit to use the server setting
  detect_faces?: boolean;
}

export interface HealthResponse {
  status: string;
  version: string;
  predictor: string; // "heuristic", "torch" or "keras"
}

export interface ModelInfo {
  predictor: string;
  num_frames: number;
  frame_size: number;
  normalize_mode: string;
  detect_faces: boolean;
  aggregation: string;
  topk_fraction: number;
  decision_threshold: number;
}
