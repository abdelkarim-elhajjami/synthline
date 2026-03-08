export type FMNodeType = 'and' | 'alt' | 'or' | 'feature';
export type OrGroupMode = 'split' | 'combine';

export interface FMAttribute {
    name: string;
    type: string;
    unit: string;
}

export interface FMNode {
    id: string;
    name: string;
    path: string;
    node_type: FMNodeType;
    group_type: FMNodeType;
    mandatory: boolean;
    abstract: boolean;
    parent_id?: string | null;
    depth: number;
    attributes: FMAttribute[];
    children: FMNode[];
}

export interface FMIndexEntry {
    id: string;
    name: string;
    path: string;
    node_type: FMNodeType;
    group_type: FMNodeType;
    mandatory: boolean;
    abstract: boolean;
    parent_id?: string | null;
    depth: number;
    attributes: FMAttribute[];
    children_ids: string[];
    is_string_feature: boolean;
}

export interface FMConstraint {
    operator: string;
    operands: FMConstraint[];
    variable?: string | null;
}

export interface FMDocument {
    source_path: string;
    root: FMNode;
    artefact_type: string;
    index: Record<string, FMIndexEntry>;
    constraints: FMConstraint[];
}

export interface FMConfiguration {
    selected_options: Record<string, string[]>;
    string_values: Record<string, string[]>;
    selected_features: string[];
    or_group_mode: Record<string, OrGroupMode>;
}

export interface FormData {
    classification_label?: string;
    classification_label_def?: string;

    fm_configuration: FMConfiguration;

    llm: string;
    temperature: number;
    top_p: number;
    samples_per_prompt: number;
    prompt_approach: string;
    pace_iterations: number;
    pace_actors: number;
    pace_candidates: number;
    pace_alpha: number;

    align_verify: boolean;
    align_threshold: number;

    total_samples: number;
}

export interface Sample {
    [key: string]: unknown;
}

export interface ScoreStats {
    count: number;
    min: number | null;
    mean: number | null;
    max: number | null;
}

export interface PromptEntry {
    prompt: string;
    features: Record<string, string | string[]>;
    samples_produced: number;
    optimized: boolean;
    pace_score?: number;
    pace?: {
        iterations: number;
        actors: number;
        candidates: number;
        alpha: number;
    };
}

export interface AlignmentVerificationBlock {
    alignment_threshold: number;
    max_retries: number;
    attempts_used: number;
    requested_samples: number;
    accepted_samples: number;
    alignment_deficit: number;
    termination_reason: string;
    scores: { accepted: ScoreStats; rejected: ScoreStats };
    total_generated_across_retries: number;
    accepted_per_attempt: number[];
    attempt_trace: Array<{
        attempt: number;
        pending_in: number;
        accepted: number;
        rejected: number;
        accepted_added: number;
        duration_ms: number;
    }>;
}

export interface GenerationMetadata {
    run_id: string;
    timestamp_utc: string;
    llm: string;
    temperature: number;
    top_p: number;
    samples_requested: number;
    samples_produced: number;
    verify: boolean;
    verify_threshold: number | null;
    optimized: boolean;
    prompt_approach: string;
    duration_seconds: number;
    alignment_verification: false | AlignmentVerificationBlock;
    prompts: PromptEntry[];
    warnings?: string[];
}

export interface GenerationOutput {
    samples: Sample[];
    output_path?: string;
    output_content: string;
    metadata: GenerationMetadata;
}

export interface AtomicPrompt {
    config: Record<string, unknown>;
    prompt: string;
    score: number;
}
