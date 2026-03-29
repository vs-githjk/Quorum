import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import { QuorumCard, SOURCE_META } from "../lib/types";
import CardViz from "./CardViz";

interface Props {
  card: QuorumCard;
  onComplete: () => void;
}

type Stage = "pill" | "morphing" | "card";

export default function QuorumIsland({ card, onComplete }: Props) {
  const [stage, setStage] = useState<Stage>("pill");
  const [currentStep, setCurrentStep] = useState(0);
  const steps = card.processing_steps;
  const meta = SOURCE_META[card.source];

  useEffect(() => {
    setStage("pill");
    setCurrentStep(0);
    const timers: ReturnType<typeof setTimeout>[] = [];
    steps.forEach((_, i) => {
      if (i > 0) {
        timers.push(setTimeout(() => setCurrentStep(i), i * 1400));
      }
    });
    timers.push(setTimeout(() => setStage("morphing"), steps.length * 1400 + 400));
    timers.push(setTimeout(() => setStage("card"), steps.length * 1400 + 1200));
    timers.push(setTimeout(() => onComplete(), steps.length * 1400 + 2000));
    return () => timers.forEach(clearTimeout);
  }, [card.id]);

  const w = stage === "pill" ? 420 : 720;
  const h = stage === "pill" ? 60 : stage === "morphing" ? 120 : "auto";
  const r = stage === "pill" ? 40 : stage === "morphing" ? 24 : 20;
  const bg = stage === "card" ? "#111" : "#161616";
  const bdr = stage === "card" ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(255,255,255,0.06)";
  const shadow = stage === "card" ? "0 8px 60px rgba(29,158,117,0.08), 0 2px 20px rgba(0,0,0,0.4)" : "none";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1, width: w, height: h, borderRadius: r }}
      transition={{ duration: 0.6, ease: [0.25, 1, 0.5, 1] }}
      style={{ background: bg, border: bdr, boxShadow: shadow, overflow: "hidden", maxWidth: "90%" }}
    >
      {stage === "pill" && (
        <div className="flex items-center px-6 gap-4 h-[60px]">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
            className="w-5 h-5 shrink-0 rounded-full border-2 border-white/[0.08] border-t-white/80"
          />
          <div className="flex-1 overflow-hidden">
            <AnimatePresence mode="wait">
              <motion.p
                key={currentStep}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.3 }}
                className="text-[14px] text-white/70 font-light whitespace-nowrap"
                style={{ fontFamily: "'Space Grotesk', sans-serif" }}
              >
                {steps[currentStep]}
              </motion.p>
            </AnimatePresence>
          </div>
          <div className="flex gap-1.5 shrink-0">
            {steps.map((_, i) => (
              <div
                key={i}
                className={`w-1.5 h-1.5 rounded-full transition-all duration-300 ${i <= currentStep ? "bg-emerald-500" : "bg-white/10"}`}
              />
            ))}
          </div>
        </div>
      )}

      {stage === "morphing" && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="flex items-center gap-4 px-8 h-[120px]"
        >
          <div
            className="w-12 h-12 rounded-xl flex items-center justify-center text-[15px] font-medium shrink-0"
            style={{ backgroundColor: meta.color + "18", color: meta.color }}
          >
            {meta.icon}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-[11px] text-white/25 uppercase tracking-widest mb-1">{meta.label}</p>
            <p
              className="text-[18px] font-light text-white/80 truncate tracking-tight"
              style={{ fontFamily: "'Space Grotesk', sans-serif" }}
            >
              {card.title}
            </p>
          </div>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
            className="w-5 h-5 shrink-0 rounded-full border-2 border-white/[0.08] border-t-emerald-500"
          />
        </motion.div>
      )}

      {stage === "card" && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="p-10"
        >
          <div className="flex items-start justify-between mb-8">
            <div className="flex items-center gap-4">
              <div
                className="w-14 h-14 rounded-xl flex items-center justify-center text-[16px] font-medium"
                style={{ backgroundColor: meta.color + "18", color: meta.color }}
              >
                {meta.icon}
              </div>
              <div>
                <p className="text-[11px] text-white/25 uppercase tracking-widest">{meta.label}</p>
                {card.badge && (
                  <span
                    className="inline-block text-[11px] font-medium px-2.5 py-0.5 rounded-md mt-1.5"
                    style={{ backgroundColor: meta.color + "18", color: meta.color }}
                  >
                    {card.badge}
                  </span>
                )}
              </div>
            </div>
            <a
              href={card.url}
              target="_blank"
              rel="noopener noreferrer"
              className="w-11 h-11 rounded-full bg-white/[0.05] border border-white/[0.08] flex items-center justify-center hover:bg-white/[0.1] transition-colors cursor-pointer group"
            >
              <ArrowUpRight className="w-[18px] h-[18px] text-white/40 group-hover:text-white/70 transition-colors" />
            </a>
          </div>
          <h2
            className="text-[28px] font-light text-white/90 leading-snug mb-6 tracking-tight"
            style={{ fontFamily: "'Space Grotesk', sans-serif" }}
          >
            {card.title}
          </h2>
          {card.visualization && <CardViz viz={card.visualization} />}
          <p className="text-[15px] text-white/45 leading-[1.7] font-light mb-8">
            {card.summary}
          </p>
          <div className="pt-5 border-t border-white/[0.06]">
            <p className="text-[12px] text-white/20 font-light italic leading-relaxed">
              &quot;{card.triggered_by}&quot;
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}