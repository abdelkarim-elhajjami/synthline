import { useEffect, useRef, useState } from "react";

const INITIAL_RECONNECT_MS = 1000;
const MAX_RECONNECT_MS = 30_000;
const BACKOFF_FACTOR = 2;

interface UseSocketLifecycleParams {
    connectionId: string;
    onMessage: (data: unknown) => void;
}

export function useSocketLifecycle({ connectionId, onMessage }: UseSocketLifecycleParams): boolean {
    const [wsReady, setWsReady] = useState(false);
    const onMessageRef = useRef(onMessage);

    useEffect(() => {
        onMessageRef.current = onMessage;
    }, [onMessage]);

    useEffect(() => {
        let cancelled = false;
        let ws: WebSocket | null = null;
        let reconnectDelay = INITIAL_RECONNECT_MS;
        let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

        function connect() {
            if (cancelled) return;

            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const wsHost = window.location.host;
            const wsUrl = `${protocol}//${wsHost}/ws/${connectionId}`;
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                reconnectDelay = INITIAL_RECONNECT_MS;
                setWsReady(true);
            };

            ws.onclose = () => {
                setWsReady(false);
                scheduleReconnect();
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    onMessageRef.current(data);
                } catch (error) {
                    console.error("WebSocket message parsing error:", error);
                }
            };
        }

        function scheduleReconnect() {
            if (cancelled) return;
            reconnectTimer = setTimeout(() => {
                reconnectTimer = null;
                connect();
            }, reconnectDelay);
            reconnectDelay = Math.min(reconnectDelay * BACKOFF_FACTOR, MAX_RECONNECT_MS);
        }

        connect();

        return () => {
            cancelled = true;
            if (reconnectTimer !== null) {
                clearTimeout(reconnectTimer);
            }
            setWsReady(false);
            ws?.close();
        };
    }, [connectionId]);

    return wsReady;
}
