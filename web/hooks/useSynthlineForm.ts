import { useState, useCallback } from "react";
import { FormData } from "@/app/types";

const initialFormState: FormData = {
    classification_label: "",
    classification_label_def: "",
    fm_configuration: {
        selected_options: {},
        string_values: {},
        selected_features: [],
        or_group_mode: {},
    },

    llm: "",
    temperature: 1.0,
    top_p: 1.0,
    samples_per_prompt: 5,
    prompt_approach: "Default",
    pace_iterations: 3,
    pace_actors: 4,
    pace_candidates: 2,
    pace_alpha: 0.5,

    align_verify: false,
    align_threshold: 0.5,

    total_samples: 10
};

export function useSynthlineForm() {
    const [formData, setFormData] = useState<FormData>(initialFormState);

    const handleInputChange = useCallback(<K extends keyof FormData>(field: K, value: FormData[K]) => {
        setFormData(prev => ({ ...prev, [field]: value }));
    }, []);

    return {
        formData,
        handleInputChange
    };
}
