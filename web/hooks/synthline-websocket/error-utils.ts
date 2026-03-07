export function toUserFriendlyError(rawErrorMessage: string): string {
    let message = rawErrorMessage.trim();
    message = message.replace(/^(Optimization|Generation)\s+error:\s*/i, "");
    message = message.replace(
        /^LLM\s+(rate-limited|quota exceeded|provider request failed|provider authorization failed|provider request timed out):\s*/i,
        "",
    );

    const singleQuotedMessage =
        message.match(/'message':\s*'(.+?)'\s*,\s*'type'/) ||
        message.match(/'message':\s*'(.+?)'/);
    if (singleQuotedMessage?.[1]) {
        return singleQuotedMessage[1].trim();
    }

    const doubleQuotedMessage =
        message.match(/"message":\s*"(.+?)"\s*,\s*"type"/) ||
        message.match(/"message":\s*"(.+?)"/);
    if (doubleQuotedMessage?.[1]) {
        return doubleQuotedMessage[1].trim();
    }

    return message;
}
