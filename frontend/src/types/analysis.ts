export type Mode = "general" | "ab_test" | "power_analysis";

export interface PgCreds {
  host: string; port: string; dbname: string; user: string; password: string;
}

export interface Sample {
  name: string; label: string; domain: string; icon: string; mode: string; suggested_task: string;
}

export const MODE_META: Record<Mode, { heading: string; sub: string; placeholder: string; accent: string; badge: string }> = {
  general: {
    heading:     "What do you want to explore?",
    sub:         "Describe what you're looking for — patterns, trends, correlations, anomalies.",
    placeholder: "e.g. What are the main patterns in this data? Which customers are at risk? What's driving delays?",
    accent:      "#cba6f7",
    badge:       "💡 Explore & Understand",
  },
  ab_test: {
    heading:     "Describe your experiment",
    sub:         "Tell DataPilot about your A/B test — it will run the full statistical analysis.",
    placeholder: "e.g. Did the new checkout flow increase revenue? Did Drug A improve recovery vs placebo? Which segments benefited most?",
    accent:      "#89b4fa",
    badge:       "🧪 Interpret Results",
  },
  power_analysis: {
    heading:     "Design your experiment",
    sub:         "Describe what you want to test — DataPilot will calculate sample size, runtime, and sensitivity.",
    placeholder: "e.g. How many users do I need to detect a 5% lift in DAU? How long should I run this experiment to detect a 10% change?",
    accent:      "#89b4fa",
    badge:       "📐 Design Experiment",
  },
};
