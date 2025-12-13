export interface FormData {
    label: string;
    label_definition: string;
    domain: string | string[];
    language: string | string[];
    output_format: "CSV" | "JSON";
    temperature: number;
    top_p: number;
    total_samples: number;
    samples_per_prompt: number;
    llm: string;
    specification_format: string | string[];
    specification_level: string | string[];
    stakeholder: string | string[];
    prompt_approach: string;
    pace_iterations: number;
    pace_actors: number;
    pace_candidates: number;
}

export interface Sample {
    text: string;
    label: string;
    domain: string;
    language: string;
}

export interface Results {
    samples: Sample[];
    output_path?: string;
    output_content: string;
    output_format: string;
    fewer_samples_received?: boolean;
}

export interface AtomicPrompt {
    config: {
        label: string;
        label_definition: string;
        specification_format: string | string[];
        specification_level: string | string[];
        stakeholder: string | string[];
        domain: string | string[];
        language: string | string[];
        samples_per_prompt: number;
    };
    prompt: string;
    score: number;
}
