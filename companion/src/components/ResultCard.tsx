import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import { QuorumCard, SOURCE_META } from "../lib/types";
import CardViz from "./CardViz";

interface Props {
  card: QuorumCard;
}

export default function ResultCard({ card }: Props) {
  const meta = SOURCE_META[card.source];

  return (
    <motion.div
      initial={{ opacity: 0, y: -40, scale: 0.92 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.6, ease: [0.25, 1, 0.5, 1] }}
      className="w-full max-w-[720px] min-h-[480px] bg-[#111] border border-white/[0.07] rounded-2xl overflow-hidden flex flex-col"
    >
      <div className="p-10 flex-1 flex flex-col">
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

        <p className="text-[15px] text-white/45 leading-[1.7] font-light mb-8 flex-1">
          {card.summary}
        </p>

        <div className="pt-5 border-t border-white/[0.06] mt-auto">
          <p className="text-[12px] text-white/20 font-light italic leading-relaxed">
            "{card.triggered_by}"
          </p>
        </div>
      </div>
    </motion.div>
  );
}