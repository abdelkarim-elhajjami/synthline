"use client"

import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import { Key } from "lucide-react";
import { useSynthline } from '@/context/SynthlineContext';

export function ApiKeySettings() {
    const { apiKeys, setApiKeys, isGenerating, isOptimizingPrompt } = useSynthline();
    const [isOpen, setIsOpen] = useState(false);
    const [localKeys, setLocalKeys] = useState(apiKeys);

    const handleOpenChange = (open: boolean) => {
        if (open) {
            setLocalKeys(apiKeys);
        }
        setIsOpen(open);
    };

    const handleSave = () => {
        setApiKeys(localKeys);
        setIsOpen(false);
    };

    return (
        <Dialog open={isOpen} onOpenChange={handleOpenChange}>
            <DialogTrigger asChild>
                <Button
                    variant="outline"
                    className="elegant-input text-[var(--foreground)] hover:border-[var(--purple)] gap-2"
                    disabled={isGenerating || isOptimizingPrompt}
                    title="Configure API Keys"
                >
                    <Key className="h-4 w-4" />
                    <span className="hidden sm:inline">API Keys</span>
                </Button>
            </DialogTrigger>
            <DialogContent className="bg-white border-[var(--card-border)] sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2 text-[var(--foreground)]">
                        <Key className="h-5 w-5 text-[var(--purple)]" />
                        API Key Configuration
                    </DialogTitle>
                    <DialogDescription className="text-[var(--foreground-muted)]">
                        Provide your own API keys. They are stored only in your browser&apos;s memory for this session.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                    <div className="space-y-2">
                        <Label htmlFor="openai" className="text-[var(--foreground)]">
                            OpenAI API Key
                        </Label>
                        <Input
                            id="openai"
                            type="password"
                            placeholder="sk-..."
                            className="elegant-input"
                            value={localKeys.openai || ''}
                            onChange={(e) => setLocalKeys({ ...localKeys, openai: e.target.value })}
                        />
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="openrouter" className="text-[var(--foreground)]">
                            OpenRouter API Key
                        </Label>
                        <Input
                            id="openrouter"
                            type="password"
                            placeholder="sk-or-..."
                            className="elegant-input"
                            value={localKeys.openrouter || ''}
                            onChange={(e) => setLocalKeys({ ...localKeys, openrouter: e.target.value })}
                        />
                    </div>
                </div>

                <DialogFooter>
                    <Button variant="ghost" onClick={() => setIsOpen(false)} className="text-[var(--foreground-muted)]">Cancel</Button>
                    <Button type="submit" onClick={handleSave} className="bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white">Save</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
