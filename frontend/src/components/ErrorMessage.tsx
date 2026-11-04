interface Props {
  message: string;
  onRetry?: () => void;
}
// this my error page
export default function ErrorMessage({ message, onRetry }: Props) {
  return (
    <div className="rounded-xl bg-red-100 p-4 text-red-700" role="alert">
      <p className="font-medium">Something went wrong</p>
      <p className="mt-1 text-sm">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 rounded-lg bg-indigo-600 px-4 py-2 text-sm text-white hover:bg-indigo-700"
        >
          Try again
        </button>
      )}
    </div>
  );
}