/**
 * Inline SVG icons, one stroke weight (1.5). Decorative: every use sits next
 * to a word, so each icon is aria-hidden.
 */
import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 16, children, ...props }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {children}
    </svg>
  );
}

export const CheckIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3 8.5l3.2 3.2L13 4.8" />
  </Svg>
);
export const CrossIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 4l8 8M12 4l-8 8" />
  </Svg>
);
export const RunningIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="5.5" strokeDasharray="20 8" />
  </Svg>
);
export const WarnIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M8 2.2 14.3 13.5H1.7Z" />
    <path d="M8 6.5v3M8 11.6v.2" />
  </Svg>
);
export const InfoIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M8 5v3.5M8 10.8v.2" />
  </Svg>
);
export const DashIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M5.2 8h5.6" />
  </Svg>
);
export const ChevronIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 6l4 4 4-4" />
  </Svg>
);
export const CopyIcon = (p: IconProps) => (
  <Svg {...p}>
    <rect x="5.5" y="5.5" width="8" height="8.5" rx="1.5" />
    <path d="M3 10.5V3.5A1.5 1.5 0 0 1 4.5 2H10" />
  </Svg>
);
export const PaperclipIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M10.7 4.9 5.9 9.7a1.4 1.4 0 0 0 2 2l5-5a2.7 2.7 0 0 0-3.8-3.8L4.1 7.9a3.9 3.9 0 0 0 5.5 5.5" />
  </Svg>
);
export const FileIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 1.8h5.2L12 4.6v9.6H4Z" />
    <path d="M9 1.8v3h3" />
  </Svg>
);
export const PlusIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M8 3v10M3 8h10" />
  </Svg>
);
export const MoonIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M13 9.5A5.5 5.5 0 0 1 6.5 3a5.5 5.5 0 1 0 6.5 6.5Z" />
  </Svg>
);
export const SunIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="3" />
    <path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" />
  </Svg>
);
export const MenuIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" />
  </Svg>
);
export const AlertIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="6.2" />
    <path d="M8 4.8v4M8 11v.2" />
  </Svg>
);

/** The HoloDeck mark: a ruled page with a cue line. */
export function BrandMark({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true" focusable="false">
      <rect x="2.75" y="2.75" width="18.5" height="18.5" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <path d="M7.5 3v18M12 3v18M16.5 3v18" stroke="currentColor" strokeOpacity=".4" strokeWidth="1.5" />
      <path d="M3 15h18" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}
