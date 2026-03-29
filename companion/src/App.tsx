import { useState, useCallback } from "react";
import { AnimatePresence } from "framer-motion";
import { QuorumCard, AppPhase } from "./lib/types";
import { MOCK_CARDS } from "./lib/mock-data";
import IntroSequence from "./components/IntroSequence";
import Sidebar from "./components/Sidebar";
import ListeningState from "./components/ListeningState";
import QuorumIsland from "./components/QuorumIsland";

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

function App() {
  const [phase, setPhase] = useState<AppPhase>("intro");
  const [currentCard, setCurrentCard] = useState<QuorumCard | null>(null);
  const [sidebarCards, setSidebarCards] = useState<QuorumCard[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [demoRunning, setDemoRunning] = useState(false);

  const handleIntroComplete = useCallback(() => {
    setPhase("listening");
  }, []);

  const handleSidebarSelect = useCallback((card: QuorumCard) => {
    setCurrentCard(card);
    setActiveId(card.id);
    setPhase("expanded");
  }, []);

  const runDemo = useCallback(async () => {
    if (demoRunning) return;
    setDemoRunning(true);
    setSidebarCards([]);
    setCurrentCard(null);
    setActiveId(null);
    setPhase("listening");

    await wait(1500);

    for (let i = 0; i < MOCK_CARDS.length; i++) {
      const card = MOCK_CARDS[i];

      if (i > 0) {
        const prevCard = MOCK_CARDS[i - 1];
        setSidebarCards((prev) => {
          if (prev.find((c) => c.id === prevCard.id)) return prev;
          return [...prev, prevCard];
        });
        await wait(800);
      }

      setCurrentCard(card);
      setActiveId(card.id);
      setPhase("expanded");

      const totalTime = card.processing_steps.length * 1400 + 2000;

      if (i < MOCK_CARDS.length - 1) {
        await wait(totalTime + 3000);
      } else {
        await wait(totalTime);
      }
    }

    await wait(2000);
    const lastCard = MOCK_CARDS[MOCK_CARDS.length - 1];
    setSidebarCards((prev) => {
      if (prev.find((c) => c.id === lastCard.id)) return prev;
      return [...prev, lastCard];
    });

    setDemoRunning(false);
  }, [demoRunning]);

  const reset = useCallback(() => {
    setPhase("listening");
    setCurrentCard(null);
    setSidebarCards([]);
    setActiveId(null);
    setDemoRunning(false);
  }, []);

  if (phase === "intro") {
    return <IntroSequence onComplete={handleIntroComplete} />;
  }

  return (
    <div className="flex h-screen bg-[#0a0a0a]">
      <Sidebar
        cards={sidebarCards}
        activeId={activeId}
        onSelect={handleSidebarSelect}
      />
      <div className="flex-1 flex flex-col min-w-0">
        <div className="px-8 py-5 border-b border-white/[0.04] flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span
              className="text-[16px] font-light text-white/70 tracking-tight"
              style={{ fontFamily: "'Space Grotesk', sans-serif" }}
            >
              Quorum
            </span>
          </div>
          <span className="text-[11px] font-light px-4 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400/80 tracking-wide">
            Listening
          </span>
        </div>

        <div className="flex-1 flex flex-col items-center justify-center relative overflow-hidden px-8">
          {phase === "listening" && !currentCard && (
            <ListeningState />
          )}

          {phase === "expanded" && currentCard && (
            <QuorumIsland
              key={currentCard.id}
              card={currentCard}
              onComplete={() => { }}
            />
          )}

          <div className="absolute bottom-6 right-8 flex gap-3 z-20">
            <button
              onClick={runDemo}
              disabled={demoRunning}
              className="text-[12px] px-5 py-2 rounded-xl border border-white/[0.08] bg-white/[0.03] text-white/40 hover:text-white/60 hover:bg-white/[0.06] transition-all disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed font-light tracking-wide"
              style={{ fontFamily: "'Space Grotesk', sans-serif" }}
            >
              {demoRunning ? "Running..." : "Run demo"}
            </button>
            <button
              onClick={reset}
              className="text-[12px] px-5 py-2 rounded-xl border border-white/[0.08] bg-white/[0.03] text-white/40 hover:text-white/60 hover:bg-white/[0.06] transition-all cursor-pointer font-light tracking-wide"
              style={{ fontFamily: "'Space Grotesk', sans-serif" }}
            >
              Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;