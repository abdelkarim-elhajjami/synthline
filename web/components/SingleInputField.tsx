"use client"

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { FormData } from "@/app/types";

interface SingleInputFieldProps {
    fieldName: keyof FormData;
    placeholder: string;
    label: string;
    value: string;
    onInputChange: <K extends keyof FormData>(field: K, value: FormData[K]) => void;
    disabled?: boolean;
}

export const SingleInputField = ({
    fieldName,
    placeholder,
    label,
    value,
    onInputChange,
    disabled
}: SingleInputFieldProps) => {
    const [inputValue, setInputValue] = useState("");
    const hasValue = value && value.trim() !== "";

    const setValue = () => {
        if (inputValue.trim()) {
            onInputChange(fieldName, inputValue.trim() as unknown as FormData[typeof fieldName]);
            setInputValue("");
        }
    };

    const clearValue = () => {
        onInputChange(fieldName, "" as unknown as FormData[typeof fieldName]);
    };

    return (
        <div className="space-y-3">
            <Label className="text-[var(--foreground)] text-sm font-medium">{label}</Label>
            {!hasValue ? (
                <div className="flex gap-2">
                    <Input
                        type="text"
                        placeholder={placeholder}
                        className="elegant-input"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), setValue())}
                        disabled={disabled}
                    />
                    <Button
                        onClick={setValue}
                        disabled={disabled || !inputValue.trim()}
                        className="bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white rounded-lg transition-all shrink-0"
                    >
                        Set
                    </Button>
                </div>
            ) : (
                <div className="flex flex-wrap gap-2">
                    <div className={`bg-[var(--purple)] text-white rounded-lg px-3 py-1.5 flex items-center gap-2 text-sm ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
                        <span>{value}</span>
                        <button
                            className="text-white/70 hover:text-white transition-colors"
                            onClick={clearValue}
                            disabled={disabled}
                        >
                            ×
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};
