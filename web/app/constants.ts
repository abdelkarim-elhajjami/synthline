import { FormData } from "./types";

export const SPECIFICATION_FORMATS = ["NL", "Constrained NL", "Use Case", "User Story"];
export const SPECIFICATION_LEVELS = ["High", "Detailed"];
export const STAKEHOLDERS = ["End Users", "Business Managers", "Developers", "Regulatory Bodies"];
export const DEFAULT_MODELS = [
    {
        label: "Local",
        items: [
            { value: "ollama/ministral-3:14b", label: "Ministral 3 14B (Ollama)" }
        ]
    }
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
