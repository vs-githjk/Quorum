import { QuorumCard } from "../lib/types";
import SidebarTile from "./SidebarTile";

interface Props {
  cards: QuorumCard[];
  activeId: string | null;
  onSelect: (card: QuorumCard) => void;
}

export default function Sidebar({ cards, activeId, onSelect }: Props) {
  return (
    <div className="w-[280px] bg-[#0d0d0d] border-r border-white/[0.05] flex flex-col overflow-hidden shrink-0">
      <div className="px-6 pt-6 pb-5 border-b border-white/[0.04]">
        <div className="flex items-center gap-2.5">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span
            className="text-[15px] font-light text-white/80 tracking-tight"
            style={{ fontFamily: "'Space Grotesk', sans-serif" }}
          >
            Quorum
          </span>
        </div>
        <p className="text-[11px] text-white/20 mt-2 font-light">History</p>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3">
        {cards.length === 0 && (
          <p className="text-[12px] text-white/15 font-light px-2 py-4">
            Cards will appear here
          </p>
        )}
        {cards.map((card) => (
          <SidebarTile
            key={card.id}
            card={card}
            isActive={activeId === card.id}
            onClick={() => onSelect(card)}
          />
        ))}
      </div>
    </div>
  );
}