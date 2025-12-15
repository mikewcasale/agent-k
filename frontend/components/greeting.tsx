import { motion } from "framer-motion";
import { Sparkles, Target, Beaker, Dna, Send } from "lucide-react";

export const Greeting = () => {
  return (
    <div
      className="mx-auto mt-4 flex size-full max-w-3xl flex-col justify-center px-4 md:mt-16 md:px-8"
      key="overview"
    >
      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-3 font-semibold text-2xl md:text-3xl"
        exit={{ opacity: 0, y: 10 }}
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.4 }}
      >
        <Sparkles className="size-7 text-blue-500" />
        <span className="bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
          AGENT-K
        </span>
      </motion.div>
      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="mt-1 text-lg text-zinc-500 md:text-xl"
        exit={{ opacity: 0, y: 10 }}
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.5 }}
      >
        Autonomous Multi-Agent Kaggle Competition System
      </motion.div>
      <motion.p
        animate={{ opacity: 1, y: 0 }}
        className="mt-4 max-w-lg text-sm text-zinc-400"
        exit={{ opacity: 0, y: 10 }}
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.6 }}
      >
        Discover, research, prototype, evolve, and submit winning solutions to Kaggle competitions.
      </motion.p>
      
      {/* Phase indicators */}
      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="mt-6 flex flex-wrap gap-3"
        exit={{ opacity: 0, y: 10 }}
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.7 }}
      >
        <PhaseIndicator icon={Target} label="Discovery" color="blue" />
        <PhaseIndicator icon={Beaker} label="Research" color="emerald" />
        <PhaseIndicator icon={Dna} label="Evolution" color="violet" />
        <PhaseIndicator icon={Send} label="Submission" color="amber" />
      </motion.div>
    </div>
  );
};

function PhaseIndicator({ 
  icon: Icon, 
  label, 
  color 
}: { 
  icon: React.ElementType; 
  label: string; 
  color: string;
}) {
  const colorClasses: Record<string, string> = {
    blue: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
    emerald: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
    violet: "bg-violet-500/10 text-violet-600 dark:text-violet-400",
    amber: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  };
  
  return (
    <div className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium ${colorClasses[color]}`}>
      <Icon className="size-3.5" />
      {label}
    </div>
  );
}
