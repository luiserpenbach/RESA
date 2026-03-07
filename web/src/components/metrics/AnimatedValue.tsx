import { useEffect, useRef, useState } from "react";

interface AnimatedValueProps {
  value: number | null | undefined;
  decimals?: number;
  fallback?: string;
}

export function AnimatedValue({
  value,
  decimals = 1,
  fallback = "—",
}: AnimatedValueProps) {
  const [display, setDisplay] = useState<number | null>(value ?? null);
  const prevRef = useRef<number | null>(value ?? null);
  const rafRef = useRef<number>();

  useEffect(() => {
    if (value == null) {
      setDisplay(null);
      prevRef.current = null;
      return;
    }

    const from = prevRef.current ?? value;
    const to = value;

    if (from === to) return;

    const duration = 480;
    const startTime = performance.now();

    function tick(now: number) {
      const t = Math.min((now - startTime) / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(from + (to - from) * eased);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        prevRef.current = to;
        setDisplay(to);
      }
    }

    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [value]);

  if (display == null) return <>{fallback}</>;
  return <>{display.toFixed(decimals)}</>;
}
