import { motion } from "framer-motion";

export default function ListeningState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.6 }}
      className="flex flex-col items-center gap-6"
    >
      <motion.div
        animate={{ scale: [1, 1.15, 1], opacity: [0.3, 0.8, 0.3] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        className="w-3 h-3 rounded-full bg-emerald-500"
      />
      <p
        className="text-lg text-white/30 font-light tracking-wide"
        style={{ fontFamily: "'Space Grotesk', sans-serif" }}
      >
        Waiting for a request
      </p>
    </motion.div>
  );
}