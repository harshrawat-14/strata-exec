interface DemoAlertProps {
  message: string;
  command: string | null;
}

export function DemoAlert({ message, command }: DemoAlertProps) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-lg bg-amber-950/40 border border-amber-700/50 text-amber-300 text-sm">
      <span className="text-amber-400 mt-0.5">⚠</span>
      <div>
        <p className="font-medium">{message} — Demo Mode</p>
        <p className="text-amber-400/70 text-xs mt-1">
          This feature is available when running locally.
        </p>
        {command && (
          <code className="block mt-2 px-2 py-1 rounded bg-slate-900/50 text-amber-200 text-xs font-mono break-all">
            {command}
          </code>
        )}
      </div>
    </div>
  );
}
