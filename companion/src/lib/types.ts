export type SourceType = "github" | "notion" | "slack" | "asana" | "chart" | "meeting";
export type EventType = "context_surfaced" | "action_taken" | "decision_logged";

export interface VizBar {
  label: string;
  value: number;
  color: string;
}

export interface CardVisualization {
  type: "bar_chart" | "line_chart" | "donut";
  data: VizBar[];
  caption: string;
}

export interface QuorumCard {
  id: string;
  event_type: EventType;
  source: SourceType;
  title: string;
  summary: string;
  url: string;
  triggered_by: string;
  timestamp: number;
  meeting_id: string;
  badge?: string;
  visualization?: CardVisualization;
  processing_steps: string[];
}

export type AppPhase = "intro" | "listening" | "processing" | "expanded";

export const SOURCE_META: Record<SourceType, { icon: string; label: string; color: string; bgClass: string; shortLabel: string }> = {
  github: { icon: "GH", label: "GitHub", color: "#1D9E75", bgClass: "bg-emerald-500/10", shortLabel: "Pull request" },
  notion: { icon: "N", label: "Notion", color: "#7F77DD", bgClass: "bg-violet-500/10", shortLabel: "Document" },
  slack: { icon: "S", label: "Slack", color: "#BA7517", bgClass: "bg-amber-500/10", shortLabel: "Thread" },
  asana: { icon: "A", label: "Asana", color: "#D85A30", bgClass: "bg-orange-500/10", shortLabel: "Task" },
  chart: { icon: "V", label: "Visualization", color: "#378ADD", bgClass: "bg-blue-500/10", shortLabel: "Chart" },
  meeting: { icon: "PM", label: "Past meeting", color: "#7F77DD", bgClass: "bg-violet-500/10", shortLabel: "Decision" },
};