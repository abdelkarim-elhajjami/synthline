import { useState, useCallback } from "react";
import { FormData } from "@/app/types";
import { REQUIRED_FIELDS } from "@/app/constants";

const initialFormState: FormData = {
    label: "",
    label_definition: "",
    domain: [],
    language: [],
    output_format: "CSV",
    temperature: 1.0,
    top_p: 1.0,
    total_samples: 10,
    samples_per_prompt: 5,
    llm: "gpt-4.1-nano-2025-04-14",
    specification_format: [],
    specification_level: [],
    stakeholder: [],
    prompt_approach: "Default",
    pace_iterations: 3,
    pace_actors: 4,
    pace_candidates: 2
};

export function useSynthlineForm() {
    const [formData, setFormData] = useState<FormData>(initialFormState);

    const handleInputChange = useCallback(<K extends keyof FormData>(field: K, value: FormData[K]) => {
        setFormData(prev => ({ ...prev, [field]: value }));
    }, []);

    const hasValidValue = useCallback((field: keyof FormData): boolean => {
        const value = formData[field];
        if (Array.isArray(value)) return value.length > 0;
        if (typeof value === 'string') return value.trim() !== '';
        return value !== undefined && value !== null;
    }, [formData]);

    const validateForm = useCallback((): string => {
        const missingFields = REQUIRED_FIELDS.filter(field => !hasValidValue(field));

        if (missingFields.length > 0) {
            const displayNames = missingFields.map(field =>
                field.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())
            );

            return `Please fill in: ${displayNames.join(', ')}`;
        }

        return '';
    }, [hasValidValue]);

    return {
        formData,
        handleInputChange,
        hasValidValue,
        validateForm,
        setFormData // Exposed for direct manipulation if extremely necessary, though avoid if possible
    };
}
