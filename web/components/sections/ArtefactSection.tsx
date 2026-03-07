"use client"

import { ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { SelectableTagGroup } from "@/components/SelectableTagGroup";
import { RequiredLabel } from "@/components/RequiredLabel";
import { FMNode, FormData, OrGroupMode } from "@/app/types";
import { useSynthline } from '@/context/SynthlineContext';

export function ArtefactSection() {
    const {
        root,
        fm,
        formData,
        handleInputChange,
        isGenerating,
        isOptimizingPrompt,
        loadingFM,
        getNode,
    } = useSynthline();

    const constraints = useMemo(() => fm?.constraints ?? [], [fm]);
    const fmIndex = useMemo(() => fm?.index ?? {}, [fm]);

    const [draftInputs, setDraftInputs] = useState<Record<string, string>>({});
    const isDisabled = isGenerating || isOptimizingPrompt || loadingFM;

    const fmConfig = formData.fm_configuration;

    const updateFMConfig = useCallback((next: FormData['fm_configuration']) => {
        handleInputChange('fm_configuration', next);
    }, [handleInputChange]);

    const removeBranchSelections = useCallback((branchRootId: string, cfg: FormData['fm_configuration']): FormData['fm_configuration'] => {
        const prefix = `${branchRootId}.`;

        const selected_options = Object.fromEntries(
            Object.entries(cfg.selected_options).filter(([key]) => key !== branchRootId && !key.startsWith(prefix))
        );
        const string_values = Object.fromEntries(
            Object.entries(cfg.string_values).filter(([key]) => key !== branchRootId && !key.startsWith(prefix))
        );
        const or_group_mode = Object.fromEntries(
            Object.entries(cfg.or_group_mode).filter(([key]) => key !== branchRootId && !key.startsWith(prefix))
        );
        const selected_features = cfg.selected_features.filter(
            featureId => featureId !== branchRootId && !featureId.startsWith(prefix)
        );

        return { selected_options, string_values, or_group_mode, selected_features };
    }, []);

    // --- Constraint auto-enforcement ---

    /** Collect all feature names that are "active" in the current config. */
    const getActiveFeatureNames = useCallback((cfg: FormData['fm_configuration']): Set<string> => {
        const active = new Set<string>();
        for (const [groupId, selectedIds] of Object.entries(cfg.selected_options)) {
            const groupEntry = fmIndex[groupId];
            if (groupEntry) active.add(groupEntry.name);
            for (const childId of selectedIds) {
                const childEntry = fmIndex[childId];
                if (childEntry) active.add(childEntry.name);
                else active.add(childId);  // fallback to raw value
            }
        }
        for (const featureId of cfg.selected_features) {
            const entry = fmIndex[featureId];
            if (entry) active.add(entry.name);
        }
        return active;
    }, [fmIndex]);

    /** Find a node ID by feature name in the index. */
    const findNodeIdByName = useCallback((name: string): string | null => {
        for (const [id, entry] of Object.entries(fmIndex)) {
            if (entry.name === name) return id;
        }
        return null;
    }, [fmIndex]);

    /**
     * Compute implied selections from `imp` constraints.
     * Returns a map of groupId → Set<childId> that must be auto-selected.
     */
    const computeImpliedSelections = useCallback((cfg: FormData['fm_configuration']): Map<string, Set<string>> => {
        const implied = new Map<string, Set<string>>();
        if (constraints.length === 0) return implied;

        const active = getActiveFeatureNames(cfg);

        for (const constraint of constraints) {
            if (constraint.operator !== 'imp' || constraint.operands.length !== 2) continue;

            const antecedent = constraint.operands[0];
            const consequent = constraint.operands[1];

            // Only handle simple var→var implications for now
            if (antecedent.operator !== 'var' || consequent.operator !== 'var') continue;
            if (!antecedent.variable || !consequent.variable) continue;

            if (active.has(antecedent.variable)) {
                // Find the consequent node and its parent group
                const consequentId = findNodeIdByName(consequent.variable);
                if (!consequentId) continue;

                const consequentEntry = fmIndex[consequentId];
                if (!consequentEntry?.parent_id) continue;

                const parentEntry = fmIndex[consequentEntry.parent_id];
                if (!parentEntry) continue;

                // Only auto-select for alt/or/and groups
                if (parentEntry.node_type !== 'alt' && parentEntry.node_type !== 'or' && parentEntry.node_type !== 'and') continue;

                const groupId = consequentEntry.parent_id;
                if (!implied.has(groupId)) implied.set(groupId, new Set());
                implied.get(groupId)!.add(consequentId);
            }
        }
        return implied;
    }, [constraints, fmIndex, getActiveFeatureNames, findNodeIdByName]);

    const impliedSelections = useMemo(
        () => computeImpliedSelections(fmConfig),
        [computeImpliedSelections, fmConfig]
    );

    // Apply implied selections to the config when they change
    useEffect(() => {
        if (impliedSelections.size === 0) return;

        let changed = false;
        const next = {
            ...fmConfig,
            selected_options: { ...fmConfig.selected_options },
        };

        for (const [groupId, impliedIds] of impliedSelections) {
            const current = new Set(next.selected_options[groupId] || []);
            const parentEntry = fmIndex[groupId];
            const isAlt = parentEntry?.node_type === 'alt';

            for (const impliedId of impliedIds) {
                if (!current.has(impliedId)) {
                    if (isAlt) {
                        // For alt groups: replace entire selection
                        next.selected_options[groupId] = [impliedId];
                    } else {
                        // For or groups: add to selection
                        current.add(impliedId);
                        next.selected_options[groupId] = Array.from(current);
                    }
                    changed = true;
                }
            }
        }

        if (changed) {
            updateFMConfig(next);
        }
    }, [impliedSelections]); // eslint-disable-line react-hooks/exhaustive-deps

    const toggleGroupOption = useCallback((group: FMNode, childId: string) => {
        if (isDisabled) return;

        // Prevent deselecting locked values
        const groupLocked = impliedSelections.get(group.id);
        if (groupLocked?.has(childId)) return;

        const current = fmConfig.selected_options[group.id] || [];
        const isSelected = current.includes(childId);
        let next = {
            ...fmConfig,
            selected_options: { ...fmConfig.selected_options },
            string_values: { ...fmConfig.string_values },
            or_group_mode: { ...fmConfig.or_group_mode },
            selected_features: [...fmConfig.selected_features],
        };

        if (isSelected) {
            next.selected_options[group.id] = current.filter(id => id !== childId);
            next = removeBranchSelections(childId, next);
        } else {
            // Coverage-by-design: alt/or/and all allow multi-selection in UI.
            // The resolver then atomizes ALT selections into valid one-per-variant configs.
            next.selected_options[group.id] = [...current, childId];
        }

        updateFMConfig(next);
    }, [fmConfig, isDisabled, removeBranchSelections, updateFMConfig, impliedSelections]);

    const toggleOptionalFeature = useCallback((node: FMNode) => {
        if (isDisabled) return;
        const selected = fmConfig.selected_features.includes(node.id);
        let next = {
            ...fmConfig,
            selected_options: { ...fmConfig.selected_options },
            string_values: { ...fmConfig.string_values },
            or_group_mode: { ...fmConfig.or_group_mode },
            selected_features: [...fmConfig.selected_features],
        };

        if (selected) {
            next.selected_features = next.selected_features.filter(id => id !== node.id);
            next = removeBranchSelections(node.id, next);
        } else {
            next.selected_features.push(node.id);
        }
        updateFMConfig(next);
    }, [fmConfig, isDisabled, removeBranchSelections, updateFMConfig]);

    const setGroupMode = useCallback((groupId: string, mode: OrGroupMode) => {
        if (isDisabled) return;
        updateFMConfig({
            ...fmConfig,
            or_group_mode: {
                ...fmConfig.or_group_mode,
                [groupId]: mode,
            },
        });
    }, [fmConfig, isDisabled, updateFMConfig]);

    const addStringValue = useCallback((nodeId: string) => {
        if (isDisabled) return;
        const raw = (draftInputs[nodeId] || '').trim();
        if (!raw) return;

        const parts = raw
            .split(',')
            .map(p => p.trim())
            .filter(Boolean);

        if (parts.length === 0) return;

        const current = fmConfig.string_values[nodeId] || [];
        const merged = [...current, ...parts.filter(p => !current.includes(p))];

        updateFMConfig({
            ...fmConfig,
            string_values: {
                ...fmConfig.string_values,
                [nodeId]: merged,
            },
        });

        setDraftInputs(prev => ({ ...prev, [nodeId]: '' }));
    }, [draftInputs, fmConfig, isDisabled, updateFMConfig]);

    const removeStringValue = useCallback((nodeId: string, index: number) => {
        if (isDisabled) return;
        const current = fmConfig.string_values[nodeId] || [];
        const nextValues = current.filter((_, i) => i !== index);

        updateFMConfig({
            ...fmConfig,
            string_values: {
                ...fmConfig.string_values,
                [nodeId]: nextValues,
            },
        });
    }, [fmConfig, isDisabled, updateFMConfig]);

    const renderNode = (node: FMNode, level: number = 0): ReactNode => {
        const isStringNode = node.attributes.some(attr => attr.type.toLowerCase() === 'string');
        const indentClass = level > 0 ? `pl-4 ${level === 1 ? 'border-l-2 border-[var(--input-border)]' : ''}` : '';

        if (node.node_type === 'alt' || node.node_type === 'or') {
            const selected = fmConfig.selected_options[node.id] || [];
            const options = node.children.map(child => ({ value: child.id, label: child.name }));
            const required = node.mandatory;
            const mode = fmConfig.or_group_mode[node.id] || 'split';
            const isMissing = required && selected.length === 0 && !(impliedSelections.get(node.id)?.size);
            const locked = impliedSelections.get(node.id);
            const lockedArray = locked ? Array.from(locked) : [];

            return (
                <div key={node.id} className={`space-y-3 ${indentClass} ${isMissing ? 'bg-orange-50/50 rounded-r-lg py-2 pr-2' : ''}`}>
                    <Label className="text-[var(--foreground)] text-sm font-medium">
                        <RequiredLabel required={required}>{node.name}</RequiredLabel>
                    </Label>

                    <SelectableTagGroup
                        options={options}
                        selected={selected}
                        onToggle={(value) => toggleGroupOption(node, value)}
                        disabled={isDisabled}
                        lockedValues={lockedArray}
                    />

                    {node.node_type === 'or' && selected.length >= 2 && (
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-[var(--foreground-muted)]">Selection mode:</span>
                            <Button
                                type="button"
                                variant={mode === 'split' ? 'default' : 'outline'}
                                size="sm"
                                className={mode === 'split' ? 'bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white' : 'elegant-input'}
                                onClick={() => setGroupMode(node.id, 'split')}
                                disabled={isDisabled}
                            >
                                Split
                            </Button>
                            <Button
                                type="button"
                                variant={mode === 'combine' ? 'default' : 'outline'}
                                size="sm"
                                className={mode === 'combine' ? 'bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white' : 'elegant-input'}
                                onClick={() => setGroupMode(node.id, 'combine')}
                                disabled={isDisabled}
                            >
                                Combine
                            </Button>
                        </div>
                    )}

                    {selected.map(childId => {
                        const child = getNode(childId);
                        if (!child) return null;
                        const childIsString = child.attributes.some(attr => attr.type.toLowerCase() === 'string');
                        const childHasNestedStructure = child.children.length > 0 || childIsString;
                        if (!childHasNestedStructure) {
                            return null;
                        }
                        return renderNode(child, level + 1);
                    })}
                </div>
            );
        }

        if (isStringNode) {
            const values = fmConfig.string_values[node.id] || [];
            const required = node.mandatory;
            const isMissing = required && values.length === 0;

            return (
                <div key={node.id} className={`space-y-3 ${indentClass} ${isMissing ? 'bg-orange-50/50 rounded-r-lg py-2 pr-2' : ''}`}>
                    <Label className="text-[var(--foreground)] text-sm font-medium">
                        <RequiredLabel required={required}>{node.name}</RequiredLabel>
                    </Label>
                    <div className="flex gap-2">
                        <Input
                            type="text"
                            value={draftInputs[node.id] || ''}
                            onChange={(e) => setDraftInputs(prev => ({ ...prev, [node.id]: e.target.value }))}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                    e.preventDefault();
                                    addStringValue(node.id);
                                }
                            }}
                            placeholder={`Enter ${node.name} value(s)`}
                            className="elegant-input"
                            disabled={isDisabled}
                        />
                        <Button
                            type="button"
                            onClick={() => addStringValue(node.id)}
                            disabled={isDisabled || !(draftInputs[node.id] || '').trim()}
                            className="bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white rounded-lg transition-all shrink-0"
                        >
                            Add
                        </Button>
                    </div>
                    {values.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                            {values.map((value, idx) => (
                                <div
                                    key={`${node.id}-${value}-${idx}`}
                                    className={`bg-[var(--purple)] text-white rounded-lg px-3 py-1.5 flex items-center gap-2 text-sm ${isDisabled ? 'opacity-50 pointer-events-none' : ''}`}
                                >
                                    <span>{value}</span>
                                    <button
                                        className="text-white/70 hover:text-white transition-colors"
                                        onClick={() => removeStringValue(node.id, idx)}
                                        disabled={isDisabled}
                                    >
                                        ×
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            );
        }

        if (node.node_type === 'feature' && !node.children.length && !node.mandatory) {
            const selected = fmConfig.selected_features.includes(node.id);
            return (
                <div key={node.id} className={`space-y-2 ${indentClass}`}>
                    <Label className="text-[var(--foreground)] text-sm font-medium">{node.name}</Label>
                    <SelectableTagGroup
                        options={[{ value: node.id, label: node.name }]}
                        selected={selected ? [node.id] : []}
                        onToggle={() => toggleOptionalFeature(node)}
                        disabled={isDisabled}
                    />
                </div>
            );
        }

        if (node.node_type === 'and' && node.children.length > 0 &&
            node.children.every(child =>
                child.node_type === 'feature' &&
                child.children.length === 0 &&
                !child.attributes.some(attr => attr.type.toLowerCase() === 'string')
            )) {
            const selected = fmConfig.selected_options[node.id] || [];
            const options = node.children.map(child => ({ value: child.id, label: child.name }));
            const locked = impliedSelections.get(node.id);
            const lockedArray = locked ? Array.from(locked) : [];

            return (
                <div key={node.id} className={`space-y-3 ${indentClass}`}>
                    <Label className="text-[var(--foreground)] text-sm font-medium">
                        {node.name}
                    </Label>
                    <SelectableTagGroup
                        options={options}
                        selected={selected}
                        onToggle={(value) => toggleGroupOption(node, value)}
                        disabled={isDisabled}
                        lockedValues={lockedArray}
                    />
                </div>
            );
        }

        const visibleChildren = node.children;
        if (visibleChildren.length === 0) {
            return null;
        }

        return (
            <div key={node.id} className={`space-y-4 ${indentClass}`}>
                {visibleChildren.map(child => renderNode(child, level + 1))}
            </div>
        );
    };

    if (loadingFM) {
        return <div className="p-6 text-center text-sm text-[var(--foreground-muted)]">Loading feature model...</div>;
    }

    if (!root) {
        return (
            <div className="p-6 text-center text-sm text-red-600">
                No active Feature Model loaded. Upload an FM above.
            </div>
        );
    }

    const artefactNode = root;
    const artefactTypeNode = artefactNode.children[0];
    const displayNode = artefactTypeNode || artefactNode;

    return (
        <section className="space-y-4">
            <h2 className="section-heading">{displayNode.name} Artefact</h2>
            <div className="elegant-card p-6">
                <fieldset disabled={isDisabled} className="space-y-6">
                    {displayNode.children.map(child => renderNode(child))}
                </fieldset>
            </div>
        </section>
    );
}
