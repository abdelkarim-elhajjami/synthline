interface ValidationDetail {
    msg?: unknown;
}

export async function responseErrorMessage(
    response: Response,
    fallback: string,
): Promise<string> {
    try {
        const data: unknown = await response.json();
        if (!data || typeof data !== "object") {
            return fallback;
        }

        const payload = data as Record<string, unknown>;
        for (const key of ["error", "detail", "message"]) {
            const value = payload[key];
            if (typeof value === "string" && value.trim()) {
                return value;
            }
            if (Array.isArray(value)) {
                const messages = value
                    .map((item: ValidationDetail) => item?.msg)
                    .filter((message): message is string => (
                        typeof message === "string" && message.trim().length > 0
                    ));
                if (messages.length > 0) {
                    return messages.join(" ");
                }
            }
        }
    } catch {
        // Fall back to the operation-specific message for non-JSON responses.
    }

    return fallback;
}
