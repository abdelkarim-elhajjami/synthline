"use client"

import { useState, ReactNode } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { FormData } from "@/app/types";

interface MultiInputFieldProps {
    fieldName: keyof FormData;
    placeholder: string;
    label: string | ReactNode;
    values: string | string[];
    onInputChange: <K extends keyof FormData>(field: K, value: FormData[K]) => void;
    disabled?: boolean;
}

export const MultiInputField = ({
    fieldName,
    placeholder,
    label,
    values,
    onInputChange,
    disabled
}: MultiInputFieldProps) => {
    const [inputValue, setInputValue] = useState("");

    const valuesArray = Array.isArray(values) ? values : (values ? [values] : []);

    const addValue = () => {
        if (inputValue.trim()) {
            const newValues = [...valuesArray, inputValue.trim()];
            onInputChange(fieldName, newValues as unknown as FormData[typeof fieldName]);
            setInputValue("");
        }
    };

    const removeValue = (index: number) => {
        const newValues = valuesArray.filter((_, i) => i !== index);
        onInputChange(fieldName, newValues as unknown as FormData[typeof fieldName]);
    };

    return (
        <div className="space-y-3">
            <Label className="text-[var(--foreground)] text-sm font-medium">{label}</Label>
            <div className="flex gap-2">
                <Input
                    type="text"
                    placeholder={placeholder}
                    className="elegant-input"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") {
                            e.preventDefault();
                            addValue();
                        }
                    }}
                    disabled={disabled}
                />
                <Button
                    onClick={addValue}
                    disabled={disabled || !inputValue.trim()}
                    className="bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white rounded-lg transition-all shrink-0"
                >
                    Add
                </Button>
            </div>
            {valuesArray.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                    {valuesArray.map((value, index) => (
                        <div
                            key={index}
                            className={`bg-[var(--purple)] text-white rounded-lg px-3 py-1.5 flex items-center gap-2 text-sm ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
                        >
                            <span>{value}</span>
                            <button
                                className="text-white/70 hover:text-white transition-colors"
                                onClick={() => removeValue(index)}
                                disabled={disabled}
                            >
                                ×
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
