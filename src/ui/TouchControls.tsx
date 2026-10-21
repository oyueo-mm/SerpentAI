import { useRef, useState, type MutableRefObject, type PointerEvent } from 'react';
import { t } from '../i18n';

/**
 * Mobile controls: a joystick (left) whose direction is the target heading, and a hold-to-boost button
 * (right). Screen axes equal world axes (the camera never rotates), so the joystick angle is the heading.
 */
export function TouchControls({ state }: { state: MutableRefObject<{ heading: number | null; boost: boolean }> }) {
  const [knob, setKnob] = useState<{ x: number; y: number } | null>(null);
  const [boost, setBoost] = useState(false);
  const base = useRef<HTMLDivElement>(null);
  const R = 46;

  const move = (e: PointerEvent<HTMLDivElement>) => {
    const r = base.current!.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
    const d = Math.hypot(dx, dy);
    const k = d > R ? R / d : 1;
    setKnob({ x: dx * k, y: dy * k });
    if (d > 8) state.current.heading = Math.atan2(dy, dx);
  };
  const end = () => { setKnob(null); state.current.heading = null; };
  const setB = (v: boolean) => { setBoost(v); state.current.boost = v; };

  return (
    <div className="touch-controls pointer-events-none absolute inset-x-0 bottom-3 z-20 flex items-end justify-between px-5">
      <div
        ref={base} data-testid="joystick" aria-label={t.controls.touch}
        className="pointer-events-auto relative h-28 w-28 touch-none rounded-full border border-slate-300/80 bg-white/60 shadow-sm backdrop-blur"
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); move(e); }}
        onPointerMove={(e) => { if (e.buttons || e.pointerType === 'touch') move(e); }}
        onPointerUp={end} onPointerCancel={end}
      >
        <div className="absolute left-1/2 top-1/2 h-12 w-12 rounded-full bg-sky-600/80 shadow" style={{ transform: `translate(calc(-50% + ${knob?.x ?? 0}px), calc(-50% + ${knob?.y ?? 0}px))` }} />
      </div>
      <button
        type="button" data-testid="boost-button" aria-pressed={boost}
        className={`pointer-events-auto h-20 w-20 touch-none select-none rounded-full text-sm font-bold shadow-sm ${boost ? 'bg-teal-600 text-white' : 'border border-slate-300/80 bg-white/70 text-teal-700 backdrop-blur'}`}
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setB(true); }}
        onPointerUp={() => setB(false)} onPointerCancel={() => setB(false)} onContextMenu={(e) => e.preventDefault()}
      >
        {t.hud.boost}
      </button>
    </div>
  );
}
