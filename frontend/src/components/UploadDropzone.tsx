import { useRef, useState } from "react";

const ALLOWED = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
const MAX_MB = 50;

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function UploadDropzone({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  function handle(file: File | undefined) {
    if (!file) return;
    const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setError(`Unsupported type. Use: ${ALLOWED.join(", ")}`);
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`File is larger than ${MAX_MB} MB.`);
      return;
    }
    setError(null);
    onFile(file);
  }

  function open() {
    if (!disabled) inputRef.current?.click();
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onClick={open}
        onKeyDown={(e) => e.key === "Enter" && open()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (!disabled) handle(e.dataTransfer.files[0]);
        }}
        className={`rounded-2xl border-2 border-dashed border-slate-300 p-10 text-center ${
          disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:border-indigo-600"
        }`}
      >
        <p className="font-medium text-slate-800">Drop a video here or click to browse</p>
        <p className="mt-1 text-sm text-slate-500">
          {ALLOWED.join(" ")} · up to {MAX_MB} MB
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED.join(",")}
          hidden
          onChange={(e) => {
            handle(e.target.files?.[0]);
            e.target.value = ""; // allow choosing the same file again
          }}
        />
      </div>
      {error && (
        <p className="mt-2 text-sm text-red-700" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}