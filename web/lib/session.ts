import { v4 as uuidv4 } from 'uuid';

const SESSION_KEY = 'synthline_session_id';

export function getSessionId(): string {
    if (typeof window === 'undefined') return '';

    const sessionId = localStorage.getItem(SESSION_KEY);
    if (!sessionId) {
        const newId = uuidv4();
        localStorage.setItem(SESSION_KEY, newId);
        return newId;
    }
    return sessionId;
}
