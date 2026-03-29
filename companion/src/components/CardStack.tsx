import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowUpRight, ChevronUp, ChevronDown } from "lucide-react";
import { QuorumCard, SOURCE_META } from "../lib/types";
import CardViz from "./CardViz";

interface Props {
    cards: QuorumCard[];
    activeIndex: number;
    onNavigate: (index: number) => void;
}

export default function CardStack({ cards, activeIndex, onNavigate }: Props) {
    if (cards.length === 0) return null;

    const goUp = () => {
        if (activeIndex > 0) onNavigate(activeIndex - 1);
    };

    const goDown = () => {
        if (activeIndex < cards.length - 1) onNavigate(activeIndex + 1);
    };

    return (
        <div className="flex flex-col items-center gap-6 w-full max-w-[720px]">
            {/* Up arrow */}
            {activeIndex > 0 && (
                <motion.button
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    onClick={goUp}
                    className="text-white/20 hover:text-white/50 transition-colors cursor-pointer"
                >
                    <ChevronUp className="w-6 h-6" />
                </motion.button>
            )}

            {/* Card stack */}
            <div className="relative w-full" style={{ height: 520 }}>
                {cards.map((card, i) => {
                    const offset = i - activeIndex;
                    const isActive = i === activeIndex;
                    const absOffset = Math.abs(offset);

                    if (absOffset > 2) return null;

                    return (
                        <motion.div
                            key={card.id}
                            initial={{ opacity: 0, y: 60 }}
                            animate={{
                                y: offset * 32,
                                scale: isActive ? 1 : 1 - absOffset * 0.05,
                                opacity: isActive ? 1 : 0.4 - absOffset * 0.12,
                                zIndex: 10 - absOffset,
                                filter: isActive ? "blur(0px)" : `blur(${absOffset * 1.5}px)`,
                            }}
                            transition={{ duration: 0.5, ease: [0.25, 1, 0.5, 1] }}
                            onClick={() => !isActive && onNavigate(i)}
                            className="absolute top-0 left-0 w-full cursor-pointer"
                            style={{ zIndex: 10 - absOffset }}
                        >
                            <FullCard card={card} isActive={isActive} />
                        </motion.div>
                    );
                })}
            </div>

            {/* Down arrow */}
            {activeIndex < cards.length - 1 && (
                <motion.button
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    onClick={goDown}
                    className="text-white/20 hover:text-white/50 transition-colors cursor-pointer"
                >
                    <ChevronDown className="w-6 h-6" />
                </motion.button>
            )}

            {/* Dot indicators */}
            <div className="flex gap-2">
                {cards.map((_, i) => (
                    <button
                        key={i}
                        onClick={() => onNavigate(i)}
                        className={`w-1.5 h-1.5 rounded-full transition-all duration-300 cursor-pointer ${i === activeIndex ? "bg-emerald-500 w-4" : "bg-white/15"
                            }`}
                    />
                ))}
            </div>
        </div>
    );
}

function FullCard({ card, isActive }: { card: QuorumCard; isActive: boolean }) {
    const meta = SOURCE_META[card.source];

    return (
        <div
            className={`
        w-full min-h-[480px] rounded-2xl overflow-hidden flex flex-col transition-shadow duration-500
        ${isActive
                    ? "bg-[#111] border border-white/[0.1] shadow-[0_8px_60px_rgba(29,158,117,0.08),0_2px_20px_rgba(0,0,0,0.4)]"
                    : "bg-[#111] border border-white/[0.05]"
                }
      `}
        >
            <div className="p-10 flex-1 flex flex-col">
                {/* Header */}
                <div className="flex items-start justify-between mb-8">
                    <div className="flex items-center gap-4">
                        <div
                            className="w-14 h-14 rounded-xl flex items-center justify-center text-[16px] font-medium"
                            style={{ backgroundColor: meta.color + "18", color: meta.color }}
                        >
                            {meta.icon}
                        </div>
                        <div>
                            <p className="text-[11px] text-white/25 uppercase tracking-widest">
                                {meta.label}
                            </p>
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
                        onClick={(e) => e.stopPropagation()}
                        className="w-11 h-11 rounded-full bg-white/[0.05] border border-white/[0.08] flex items-center justify-center hover:bg-white/[0.1] transition-colors cursor-pointer group"
                    >
                        <ArrowUpRight className="w-[18px] h-[18px] text-white/40 group-hover:text-white/70 transition-colors" />
                    </a>
                </div>

                {/* Title */}
                <h2
                    className="text-[28px] font-light text-white/90 leading-snug mb-6 tracking-tight"
                    style={{ fontFamily: "'Space Grotesk', sans-serif" }}
                >
                    {card.title}
                </h2>

                {/* Viz */}
                {card.visualization && <CardViz viz={card.visualization} />}

                {/* Summary */}
                <p className="text-[15px] text-white/45 leading-[1.7] font-light mb-8 flex-1">
                    {card.summary}
                </p>

                {/* Trigger */}
                <div className="pt-5 border-t border-white/[0.06] mt-auto">
                    <p className="text-[12px] text-white/20 font-light italic leading-relaxed">
                        "{card.triggered_by}"
                    </p>
                </div>
            </div>
        </div >
    );
}