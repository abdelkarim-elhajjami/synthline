import { useState, useEffect, useCallback } from 'react';
import { FMDocument, FMNode } from '@/app/types';
import { getSessionId } from '@/lib/session';

export function useFM() {
    const [fm, setFm] = useState<FMDocument | null>(null);
    const [loadingFM, setLoadingFM] = useState(true);

    const fetchFM = useCallback(async () => {
        try {
            const response = await fetch('/api/features', {
                headers: {
                    'X-Session-ID': getSessionId(),
                },
            });
            if (response.ok) {
                const data = await response.json();
                setFm(data as FMDocument);
            } else {
                console.error('Failed to fetch features:', response.statusText);
            }
        } catch (error) {
            console.error('Error fetching features:', error);
        } finally {
            setLoadingFM(false);
        }
    }, []);

    useEffect(() => {
        fetchFM();
    }, [fetchFM]);

    const uploadFM = useCallback(async (file: File) => {
        const buffer = await file.arrayBuffer();
        const response = await fetch('/api/features/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/xml',
                'x-filename': file.name,
                'X-Session-ID': getSessionId(),
            },
            body: buffer,
        });

        if (!response.ok) {
            let message = `Upload failed (${response.status})`;
            try {
                const data = await response.json();
                if (data?.detail) message = data.detail;
                if (data?.error) message = data.error;
            } catch {
            }
            throw new Error(message);
        }

        setLoadingFM(true);
        await fetchFM();
    }, [fetchFM]);

    const uploadGlossary = useCallback(async (file: File) => {
        const buffer = await file.arrayBuffer();
        const response = await fetch('/api/glossary/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'text/yaml',
                'x-filename': file.name,
                'X-Session-ID': getSessionId(),
            },
            body: buffer,
        });

        if (!response.ok) {
            let message = `Upload failed (${response.status})`;
            try {
                const data = await response.json();
                if (data?.detail) message = data.detail;
                if (data?.error) message = data.error;
            } catch {
            }
            throw new Error(message);
        }

        const data = await response.json();
        return {
            replaced: Boolean(data?.replaced),
            entries: Number(data?.entries || 0),
        };
    }, []);

    const root = fm?.root || null;
    const index = fm?.index || {};

    const getNode = useCallback((nodeId: string): FMNode | null => {
        if (!root) return null;

        const stack: FMNode[] = [root];
        while (stack.length > 0) {
            const current = stack.pop() as FMNode;
            if (current.id === nodeId) return current;
            stack.push(...current.children);
        }
        return null;
    }, [root]);

    return {
        fm,
        root,
        index,
        loadingFM,
        uploadFM,
        uploadGlossary,
        getNode,
    };
}
