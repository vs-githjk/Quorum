import { motion } from "framer-motion";
import { CardVisualization } from "../lib/types";

interface Props {
  viz: CardVisualization;
}

export default function CardViz({ viz }: Props) {
  const maxVal = Math.max(...viz.data.map((d) => d.value));

  return (
    <div className="bg-white/[0.03] border border-white/[0.05] rounded-xl p-5 mb-5">
      <p className="text-[12px] text-white/30 mb-4 tracking-wide uppercase">
        {viz.caption}
      </p>
      <div className="flex flex-col gap-3">
        {viz.data.map((bar, i) => (
          <div key={i} className="flex items-center gap-3">
            <span className="text-[13px] text-white/50 w-20 text-right shrink-0 font-light">
              {bar.label}
            </span>
            <div className="flex-1 h-6 bg-white/[0.04] rounded-md overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.round((bar.value / maxVal) * 100)}%` }}
                transition={{ duration: 1, delay: i * 0.15, ease: [0.25, 1, 0.5, 1] }}
                className="h-full rounded-md"
                style={{ backgroundColor: bar.color }}
              />
            </div>
            <span className="text-[12px] text-white/30 w-10 tabular-nums">
              {bar.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}