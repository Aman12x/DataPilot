/**
 * Minimal inline SVG icon set. All icons are 1em square, stroke-based, and
 * inherit currentColor, so they follow the text color and size of their
 * context. Decorative only: every icon sets aria-hidden.
 */

interface IconProps {
  size?: number;
  strokeWidth?: number;
  style?: React.CSSProperties;
}

function base(size: number | undefined, style: React.CSSProperties | undefined) {
  return {
    width: size ?? "1em",
    height: size ?? "1em",
    flexShrink: 0,
    verticalAlign: "-0.125em",
    ...style,
  } as React.CSSProperties;
}

export function IconCheck({ size, strokeWidth = 2, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" style={base(size, style)}>
      <path d="M3 8.5 6.5 12 13 4.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconAlert({ size, strokeWidth = 1.5, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" style={base(size, style)}>
      <path d="M8 1.8 15 14H1L8 1.8Z" stroke="currentColor" strokeWidth={strokeWidth} strokeLinejoin="round" />
      <path d="M8 6.2v3.4" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <circle cx="8" cy="11.9" r="0.9" fill="currentColor" />
    </svg>
  );
}

export function IconChevronDown({ size, strokeWidth = 2, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" style={base(size, style)}>
      <path d="M3.5 6 8 10.5 12.5 6" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconChevronUp({ size, strokeWidth = 2, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" style={base(size, style)}>
      <path d="M3.5 10 8 5.5 12.5 10" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconChevronRight({ size, strokeWidth = 2, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" style={base(size, style)}>
      <path d="M6 3.5 10.5 8 6 12.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconArrowLeft({ size, strokeWidth = 2, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" style={base(size, style)}>
      <path d="M13 8H3M7 4 3 8l4 4" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconDownload({ size, strokeWidth = 1.75, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" style={base(size, style)}>
      <path d="M8 2.5v7M5 6.5 8 9.5l3-3M2.5 12.5h11" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconPlus({ size, strokeWidth = 2, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" style={base(size, style)}>
      <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </svg>
  );
}

export function IconClose({ size, strokeWidth = 2, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" style={base(size, style)}>
      <path d="m4 4 8 8M12 4l-8 8" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </svg>
  );
}

/** Filled circle used as a status dot. */
export function IconDot({ size, style }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" style={base(size, style)}>
      <circle cx="8" cy="8" r="4" fill="currentColor" />
    </svg>
  );
}
