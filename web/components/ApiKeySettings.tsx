"use client"

import React, { useState } from 'react';
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
import { Key, ShieldCheck } from "lucide-react";
import { useSynthline } from '@/context/SynthlineContext';

export function ApiKeySettings() {
    const { apiKeys, setApiKeys, isGenerating, isOptimizingPrompt } = useSynthline();
    const [isOpen, setIsOpen] = useState(false);

    // Local state for inputs to allow cancellation
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
                    className="bg-[#1A1A1A] border-[#2A2A2A] text-white hover:bg-[#2A2A2A] hover:text-[#8A2BE2] gap-2"
                    disabled={isGenerating || isOptimizingPrompt}
                    title="Configure API Keys"
                >
                    <Key className="h-4 w-4" />
                    <span className="hidden sm:inline">Configure API Keys</span>
                </Button>
            </DialogTrigger>
            <DialogContent className="bg-[#121212] border-[#2A2A2A] text-white sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Key className="h-5 w-5 text-[#8A2BE2]" />
                        API Key Configuration
                    </DialogTitle>
                    <DialogDescription className="text-zinc-400">
                        Provide your own API keys. They are stored only in your browser&apos;s memory for this session and are never logged.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                    <div className="space-y-2">
                        <Label htmlFor="openai" className="text-right">
                            OpenAI API Key
                        </Label>
                        <Input
                            id="openai"
                            type="password"
                            placeholder="sk-..."
                            className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                            value={localKeys.openai || ''}
                            onChange={(e) => setLocalKeys({ ...localKeys, openai: e.target.value })}
                        />
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="openrouter" className="text-right">
                            OpenRouter API Key
                        </Label>
                        <Input
                            id="openrouter"
                            type="password"
                            placeholder="sk-or-..."
                            className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                            value={localKeys.openrouter || ''}
                            onChange={(e) => setLocalKeys({ ...localKeys, openrouter: e.target.value })}
                        />
                        <p className="text-[10px] text-zinc-500">Required for access to Claude 3.5, Llama 3, etc.</p>
                    </div>
                </div>

                <div className="bg-yellow-900/20 border border-yellow-800 p-3 rounded-md flex items-start gap-2 mb-4">
                    <ShieldCheck className="h-4 w-4 text-yellow-500 mt-0.5 shrink-0" />
                    <p className="text-xs text-yellow-400">
                        Keys are ephemeral (cleared on refresh) and prioritized over server-side keys.
                    </p>
                </div>

                <DialogFooter>
                    <Button variant="ghost" onClick={() => setIsOpen(false)} className="text-zinc-400 hover:text-white">Cancel</Button>
                    <Button type="submit" onClick={handleSave} className="bg-[#8A2BE2] hover:bg-opacity-80 text-white">Save Configuration</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
