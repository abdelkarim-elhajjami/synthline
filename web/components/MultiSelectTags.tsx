import { FormData } from "@/app/types";

interface MultiSelectTagsProps {
    options: string[];
    selected: string | string[];
    fieldName: keyof FormData;
    onInputChange: <K extends keyof FormData>(field: K, value: FormData[K]) => void;
    disabled: boolean;
}

export const MultiSelectTags = ({
    options,
    selected,
    fieldName,
    onInputChange,
    disabled
}: MultiSelectTagsProps) => {
    const selectedArray = Array.isArray(selected) ? selected : selected ? [selected] : [];

    const toggleSelection = (item: string) => {
        if (selectedArray.includes(item)) {
            const newSelections = selectedArray.filter(i => i !== item);
            // We know fieldName is a field that accepts array of strings based on usage,
            // but TypeScript might need reassurance.
            onInputChange(fieldName, (newSelections.length ? newSelections : []) as string[]);
        } else {
            onInputChange(fieldName, [...selectedArray, item] as string[]);
        }
    };

    return (
        <div className="flex flex-wrap gap-2 mt-2">
            {options.map((item) => {
                const isSelected = selectedArray.includes(item);
                const selectedClass = isSelected
                    ? "bg-[#8A2BE2] border-[#8A2BE2] text-white"
                    : "bg-[#1A1A1A] border-[#2A2A2A] text-zinc-300 hover:bg-[#222222] hover:border-[#8A2BE2]";

                const disabledClass = disabled ? 'opacity-50 pointer-events-none' : '';

                return (
                    <div
                        key={item}
                        className={`border rounded-lg px-4 py-2 text-sm cursor-pointer transition-colors ${selectedClass} ${disabledClass}`}
                        onClick={() => toggleSelection(item)}
                    >
                        {item} {isSelected}
                    </div>
                );
            })}
        </div>
    );
};
