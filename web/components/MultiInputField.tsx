import { useState } from "react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { FormData } from "@/app/types";

interface MultiInputFieldProps {
    fieldName: keyof FormData;
    placeholder: string;
    label: string;
    values: string | string[];
    onInputChange: <K extends keyof FormData>(field: K, value: FormData[K]) => void;
    disabled: boolean;
}

export const MultiInputField = ({
    fieldName,
    placeholder,
    label,
    values: rawValues,
    onInputChange,
    disabled
}: MultiInputFieldProps) => {
    const [inputValue, setInputValue] = useState("");
    const values = Array.isArray(rawValues) ? rawValues :
        rawValues ? [rawValues] : [];

    const addValue = () => {
        if (inputValue.trim()) {
            onInputChange(fieldName, [...values, inputValue.trim()] as string[]);
            setInputValue("");
        }
    };

    const removeValue = (value: string) => {
        const newValues = values.filter(v => v !== value);
        onInputChange(fieldName, (newValues.length ? newValues : []) as string[]);
    };

    return (
        <div className="space-y-2">
            <Label className="text-white text-base font-medium">{label}</Label>
            <div className="flex gap-2">
                <Input
                    type="text"
                    placeholder={placeholder}
                    className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addValue())}
                    disabled={disabled}
                />
                <Button
                    onClick={addValue}
                    disabled={disabled || !inputValue.trim()}
                    className="bg-[#8A2BE2] hover:bg-opacity-80 text-white"
                >
                    Add
                </Button>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
                {values.map((value, index) => (
                    <div key={index} className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg px-3 py-1 flex items-center gap-2">
                        <span className="text-white">{value}</span>
                        <button
                            className="text-zinc-400 hover:text-white"
                            onClick={() => removeValue(value)}
                            disabled={disabled}
                        >
                            ×
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
};
