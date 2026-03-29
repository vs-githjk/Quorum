import { motion } from "framer-motion";
import { QuorumCard, SOURCE_META } from "../lib/types";

interface Props {
  card: QuorumCard;
  isActive: boolean;
  onClick: () => void;
}

export default function SidebarTile({ card, isActive, onClick }: Props) {
  const meta = SOURCE_META[card.source];
  const shortTitle = card.title.length > 35 ? card.title.slice(0, 35) + "..." : card.title;

  return (
    <motion.button
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.5, ease: [0.34, 1.56, 0.64, 1] }}
      onClick={onClick}
      className={`
        w-full px-4 py-3.5 rounded-xl text-left cursor-pointer transition-all duration-200
        ${isActive
          ? "bg-white/[0.08] border border-white/[0.12]"
          : "bg-white/[0.02] border border-white/[0.04] hover:bg-white/[0.05] hover:border-white/[0.08]"
        }
      `}
    >
      <div className="flex items-center gap-3 mb-2">
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center text-[10px] font-semibold shrink-0"
          style={{ backgroundColor: meta.color + "18", color: meta.color }}
        >
          {meta.icon}
        </div>
        <span className="text-[10px] text-white/25 uppercase tracking-wider">
          {meta.shortLabel}
        </span>
        <div
          className="w-1.5 h-1.5 rounded-full ml-auto shrink-0"
          style={{ backgroundColor: meta.color }}
        />
      </div>
      <p className="text-[13px] text-white/60 font-light leading-snug">
        {shortTitle}
      </p>
    </motion.button>
  );
}