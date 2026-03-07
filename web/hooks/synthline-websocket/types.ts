export type OperationType = "generation" | "optimization";

export interface UiError {
    operation: OperationType;
    message: string;
}
