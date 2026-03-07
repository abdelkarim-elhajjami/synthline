"use client"

import { FormData } from "@/app/types";

interface MultiSelectTagsProps {
    options: { value: string; label: string }[];
    selected: string | string[];
    fieldName: keyof FormData;
    onInputChange: <K extends keyof FormData>(field: K, value: FormData[K]) => void;
    disabled?: boolean;
}

export function MultiSelectTags({ options, selected, fieldName, onInputChange, disabled }: MultiSelectTagsProps) {
    const selectedArray = Array.isArray(selected) ? selected : (selected ? [selected] : []);

    const toggleOption = (value: string) => {
        if (disabled) return;

        const newSelected = selectedArray.includes(value)
            ? selectedArray.filter(v => v !== value)
            : [...selectedArray, value];

        onInputChange(fieldName, newSelected as unknown as FormData[typeof fieldName]);
    };

    return (
        <div className="flex flex-wrap gap-2">
            {options.map(option => {
                const isSelected = selectedArray.includes(option.value);
                return (
                    <div
                        key={option.value}
                        onClick={() => toggleOption(option.value)}
                        className={`elegant-tag px-3 py-1.5 text-sm cursor-pointer ${isSelected ? 'selected' : ''} ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
                    >
                        {option.label}
                    </div>
                );
            })}
        </div>
    );
}
