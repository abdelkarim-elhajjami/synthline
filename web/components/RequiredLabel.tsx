import { ReactNode } from "react";

interface RequiredLabelProps {
    children: ReactNode;
    required?: boolean;
    className?: string;
    indicatorClassName?: string;
}

export const RequiredLabel = ({ children, required, className, indicatorClassName }: RequiredLabelProps) => {
    if (!required) return <>{children}</>;

    return (
        <span className={`flex items-center gap-1 ${className || ''}`}>
            {children}
            <span className={indicatorClassName || "text-red-500 font-bold"}>*</span>
        </span>
    );
};
