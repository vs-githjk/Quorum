import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
    onComplete: () => void;
}

export default function IntroSequence({ onComplete }: Props) {
    const [phase, setPhase] = useState(0);

    useEffect(() => {
        const t1 = setTimeout(() => setPhase(1), 2000);
        const t2 = setTimeout(() => setPhase(2), 4200);
        const t3 = setTimeout(() => onComplete(), 6400);
        return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
    }, [onComplete]);

    const texts = ["Hello.", "Welcome to Quorum.", ""];

    return (
        <div className="fixed inset-0 bg-[#0a0a0a] flex items-center justify-center z-50">
            <AnimatePresence mode="wait">
                {phase < 2 && (
                    <motion.h1
                        key={phase}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
                        className="text-5xl md:text-7xl font-light tracking-tight text-white"
                        style={{ fontFamily: "'Space Grotesk', sans-serif" }}
                    >
                        {texts[phase]}
                    </motion.h1>
                )}
            </AnimatePresence>
        </div>
    );
}