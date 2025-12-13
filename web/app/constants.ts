import { FormData } from "./types";

export const SPECIFICATION_FORMATS = ["NL", "Constrained NL", "Use Case", "User Story"];
export const SPECIFICATION_LEVELS = ["High", "Detailed"];
export const STAKEHOLDERS = ["End Users", "Business Managers", "Developers", "Regulatory Bodies"];
export const LLM_OPTIONS = [
    { value: "gpt-4.1-nano-2025-04-14", label: "gpt-4.1-nano-2025-04-14" },
    { value: "deepseek-chat", label: "deepseek-chat" },
    { value: "ollama/ministral-3:14b", label: "ollama/ministral-3:14b" }
];
export const REQUIRED_FIELDS: (keyof FormData)[] = [
    'label',
    'label_definition',
    'domain',
    'language',
    'specification_format',
    'specification_level',
    'stakeholder'
];
