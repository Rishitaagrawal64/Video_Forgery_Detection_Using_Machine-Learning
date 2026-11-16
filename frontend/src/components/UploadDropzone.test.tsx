import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import UploadDropzone from "./UploadDropzone";

function setup(props: { maxMb?: number; allowedExtensions?: string[] } = {}) {
  const onFile = vi.fn();
  const { container } = render(<UploadDropzone onFile={onFile} {...props} />);
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  return { onFile, input };
}

function pick(input: HTMLInputElement, file: File) {
  fireEvent.change(input, { target: { files: [file] } });
}

describe("UploadDropzone", () => {
  it("accepts a valid video", () => {
    const { onFile, input } = setup();
    const file = new File(["x"], "clip.mp4", { type: "video/mp4" });
    pick(input, file);
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("rejects an unsupported file type", () => {
    const { onFile, input } = setup();
    pick(input, new File(["x"], "notes.pdf", { type: "application/pdf" }));
    expect(onFile).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("Unsupported type");
  });

  it("rejects a file larger than 50 MB", () => {
    const { onFile, input } = setup();
    const big = new File(["x"], "big.mp4", { type: "video/mp4" });
    Object.defineProperty(big, "size", { value: 51 * 1024 * 1024 });
    pick(input, big);
    expect(onFile).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("50 MB");
  });

  it("uses the server's size limit when provided", () => {
    const { onFile, input } = setup({ maxMb: 10 });
    const clip = new File(["x"], "clip.mp4", { type: "video/mp4" });
    Object.defineProperty(clip, "size", { value: 11 * 1024 * 1024 });
    pick(input, clip);
    expect(onFile).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("10 MB");
  });

  it("uses the server's allowed extensions when provided", () => {
    const { onFile, input } = setup({ allowedExtensions: [".mp4"] });
    pick(input, new File(["x"], "clip.mov", { type: "video/quicktime" }));
    expect(onFile).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("Unsupported type");
  });
});
