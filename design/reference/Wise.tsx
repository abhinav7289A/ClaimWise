// Wise — the ClaimWise robot mascot. Pure SVG, no assets.
// Reuse everywhere (nav icon, chat avatars, hero, loader states) by changing `size`.
// `state` drives the face; `stroke` overrides the line color (use #8fb2d6 on dark).
import * as React from 'react';

type WiseState = 'idle' | 'thinking' | 'done' | 'error';

export function Wise({
  state = 'idle',
  size = 120,
  stroke,
  compact = false, // head-only (for small avatars / nav icon)
}: {
  state?: WiseState;
  size?: number;
  stroke?: string;
  compact?: boolean;
}) {
  const color = stroke ?? (state === 'error' ? '#b46a4d' : '#416180');
  const line = {
    stroke: color,
    strokeWidth: 2.6,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    fill: 'none',
  };
  const solid = { fill: color, stroke: 'none' };

  // Face per state
  const face = {
    idle: (
      <>
        <circle cx={44} cy={53} r={6.5} {...solid} />
        <circle cx={76} cy={53} r={6.5} {...solid} />
        <path d="M48 69 q12 10 24 0" {...line} />
      </>
    ),
    thinking: (
      <>
        <circle cx={92} cy={14} r={3} {...solid} opacity={0.4} />
        <circle cx={101} cy={7} r={4.5} {...solid} opacity={0.7} />
        <circle cx={48} cy={50} r={4} {...solid} />
        <circle cx={80} cy={50} r={4} {...solid} />
        <line x1={50} y1={70} x2={70} y2={70} {...line} />
      </>
    ),
    done: (
      <>
        <path d="M39 53 q5 -6 10 0" {...line} />
        <path d="M71 53 q5 -6 10 0" {...line} />
        <path d="M46 67 q14 12 28 0" {...line} />
      </>
    ),
    error: (
      <>
        <line x1={40} y1={48} x2={48} y2={56} {...line} />
        <line x1={48} y1={48} x2={40} y2={56} {...line} />
        <line x1={72} y1={48} x2={80} y2={56} {...line} />
        <line x1={80} y1={48} x2={72} y2={56} {...line} />
        <path d="M48 72 q12 -8 24 0" {...line} />
      </>
    ),
  }[state];

  if (compact) {
    // head only — good for 20-30px avatars / nav
    return (
      <svg viewBox="18 22 84 64" width={size} height={(size * 64) / 84}>
        <rect x={20} y={22} width={80} height={62} rx={22} {...line} />
        {state === 'idle' && (
          <>
            <circle cx={45} cy={52} r={6} {...solid} />
            <circle cx={75} cy={52} r={6} {...solid} />
            <path d="M49 68 q11 9 22 0" {...line} />
          </>
        )}
        {state !== 'idle' && face}
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 120 130" width={size} height={(size * 130) / 120}>
      {/* antenna */}
      <line x1={60} y1={4} x2={60} y2={22} {...line} />
      <circle cx={60} cy={4} r={5} {...solid} />
      {/* head */}
      <rect x={18} y={22} width={84} height={64} rx={24} {...line} />
      {face}
      {/* body */}
      <rect x={30} y={94} width={60} height={30} rx={13} {...line} />
      {state === 'done' && <path d="M46 109 l7 7 13 -14" stroke={color} strokeWidth={3} fill="none" strokeLinecap="round" strokeLinejoin="round" />}
      {/* arms */}
      <line x1={18} y1={104} x2={30} y2={104} {...line} />
      <line x1={90} y1={104} x2={102} y2={104} {...line} />
    </svg>
  );
}

/*
Typing indicator (three pulsing dots) — pair with a Wise avatar whose mouth is a flat line:

  .dot { width:7px; height:7px; border-radius:50%; background:#8fb2d6; animation:dotp 1.4s ease-in-out infinite; }
  .dot:nth-child(2){ animation-delay:.2s } .dot:nth-child(3){ animation-delay:.4s }
  @keyframes dotp { 0%,60%,100%{ opacity:.25; transform:translateY(0) } 30%{ opacity:1; transform:translateY(-3px) } }
*/
