interface Option {
    value: string;
    label: string;
}

interface SelectableTagGroupProps {
    options: Option[];
    selected: string | string[];
    onToggle: (value: string) => void;
    disabled?: boolean;
    className?: string;
    lockedValues?: string[];
}

export function SelectableTagGroup({
    options,
    selected,
    onToggle,
    disabled = false,
    className = "",
    lockedValues = []
}: SelectableTagGroupProps) {
    const selectedArray = Array.isArray(selected) ? selected : (selected ? [selected] : []);

    return (
        <div className={`flex flex-wrap gap-2 ${className}`}>
            {options.map((item) => {
                const isLocked = lockedValues.includes(item.value);
                const isSelected = isLocked || selectedArray.includes(item.value);
                return (
                    <div
                        key={item.value}
                        className={`elegant-tag px-4 py-2 text-sm ${isSelected ? 'selected' : ''} ${isLocked ? 'cursor-default' : 'cursor-pointer'} ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
                        onClick={() => !disabled && !isLocked && onToggle(item.value)}
                    >
                        {item.label}
                    </div>
                );
            })}
        </div>
    );
}
