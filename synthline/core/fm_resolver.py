"""
Resolve the user's FM selection into a list of atomic configurations.
"""
from __future__ import annotations

from typing import Any, Dict, List

from synthline.core.fm_parser import FM, FMConstraint, FMNode


class FMResolver:
    def __init__(self, fm: FM):
        self._fm = fm
        self._node_index: Dict[str, FMNode] = {node.id: node for node in fm.iter_nodes()}
        self._constraints = fm.constraints

    def resolve(self, fm_configuration: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected_options = fm_configuration.get("selected_options", {}) or {}
        string_values = fm_configuration.get("string_values", {}) or {}
        selected_features = set(fm_configuration.get("selected_features", []) or [])
        or_group_mode = fm_configuration.get("or_group_mode", {}) or {}
        allow_empty_fallback = bool(fm_configuration.get("allow_empty_fallback", False))

        context = {
            "selected_options": selected_options,
            "string_values": string_values,
            "selected_features": selected_features,
            "or_group_mode": or_group_mode,
        }

        variants = self._resolve_node(self._fm.root, context)
        cleaned_variants = [self._clean_variant(variant) for variant in variants]
        non_empty = [variant for variant in cleaned_variants if variant]

        valid_variants = [v for v in non_empty if self._satisfies_constraints(v)]

        if not valid_variants:
            if allow_empty_fallback:
                valid_variants = [{}]
            else:
                raise ValueError(
                    "No valid FM configurations after applying group semantics and cross-tree constraints. "
                    "Set `allow_empty_fallback=true` to return an empty fallback configuration."
                )

        configs: List[Dict[str, Any]] = []
        for variant in valid_variants:
            atomic = dict(variant)
            atomic["__fm_constraints__"] = self._build_constraints(variant)
            configs.append(atomic)

        return configs

    # ── Dispatcher ──────────────────────────────────────────────────────────
    def _resolve_node(self, node: FMNode, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        if node.node_type == "alt":
            return self._resolve_alt_group(node, ctx)
        if node.node_type == "or":
            return self._resolve_or_group(node, ctx)
        if node.node_type == "and":
            return self._resolve_and_group(node, ctx)
        if node.node_type == "feature":
            return self._resolve_feature(node, ctx)
        return [{}]

    # ── alt: exactly-one – each selected child produces a separate variant ──
    def _resolve_alt_group(self, node: FMNode, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected_children = self._resolve_selected_children(
            group_node=node,
            selected_raw=ctx["selected_options"].get(node.id, []) or [],
        )
        if not selected_children:
            if node.mandatory or self._is_group_explicitly_selected(node, ctx):
                return []
            return [{}]

        results: List[Dict[str, Any]] = []
        for child in selected_children:
            child_variants = self._resolve_node(child, ctx)
            for child_variant in child_variants:
                merged = {node.id: child.name}
                merged.update(child_variant)
                results.append(merged)
        return results

    # ── or: one-or-more – split (one variant per child) or combine (all together) ──
    def _resolve_or_group(self, node: FMNode, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected_children = self._resolve_selected_children(
            group_node=node,
            selected_raw=ctx["selected_options"].get(node.id, []) or [],
        )
        if not selected_children:
            if node.mandatory or self._is_group_explicitly_selected(node, ctx):
                return []
            return [{}]

        mode = ctx["or_group_mode"].get(node.id, "split")
        if mode not in {"split", "combine"}:
            raise ValueError(
                f"Unsupported or_group_mode '{mode}' for node '{node.id}'. "
                "Expected one of: split, combine."
            )
        if mode == "combine":
            return self._resolve_combine(node, selected_children, ctx)

        # Split mode — same as alt: one variant per child
        results: List[Dict[str, Any]] = []
        for child in selected_children:
            child_variants = self._resolve_node(child, ctx)
            for child_variant in child_variants:
                merged = {node.id: child.name}
                merged.update(child_variant)
                results.append(merged)
        return results

    # ── and: all-together – combine semantics, always includes mandatory children ──
    def _resolve_and_group(self, node: FMNode, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        # If explicit selections exist (tag-group UI), resolve those + mandatory.
        and_selected_raw = ctx["selected_options"].get(node.id, []) or []
        if and_selected_raw:
            selected_children = self._resolve_selected_children(node, and_selected_raw)
            # Always include mandatory children alongside explicit selections.
            selected_ids = {child.id for child in selected_children}
            for child in node.children:
                if child.mandatory and child.id not in selected_ids:
                    selected_children.append(child)
            if not selected_children:
                return [{}]
            return self._resolve_combine(node, selected_children, ctx)

        # Structural and-group (no explicit selections) — iterate active children.
        variants: List[Dict[str, Any]] = [{}]

        if node.is_string_feature:
            values = self._get_string_values(node.id, ctx["string_values"])
            if values:
                variants = [{node.id: value} for value in values]

        for child in node.children:
            if not self._is_child_active(child, ctx):
                continue
            child_variants = self._resolve_node(child, ctx)
            variants = self._merge_variant_lists(variants, child_variants)
        return variants

    # ── feature: leaf node ──
    def _resolve_feature(self, node: FMNode, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        if node.is_string_feature:
            values = self._get_string_values(node.id, ctx["string_values"])
            if values:
                return [{node.id: value} for value in values]
            if node.mandatory:
                return []
            return [{}]

        if not node.children:
            if node.mandatory or node.id in ctx["selected_features"]:
                return [{node.id: True}]
            return [{}]

        # Feature with children (rare) — walk children like a structural node.
        variants: List[Dict[str, Any]] = [{}]
        for child in node.children:
            if not self._is_child_active(child, ctx):
                continue
            child_variants = self._resolve_node(child, ctx)
            variants = self._merge_variant_lists(variants, child_variants)
        return variants

    # ── shared: combine all selected children into a single variant ──
    def _resolve_combine(
        self, node: FMNode, children: List[FMNode], ctx: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """All selected children appear together in every variant (and / or-combine)."""
        combined_label_values = [child.name for child in children]

        child_variant_lists: List[List[Dict[str, Any]]] = []
        for child in children:
            child_variant_lists.append(self._resolve_node(child, ctx))

        combinations = self._product_variants(child_variant_lists)
        results: List[Dict[str, Any]] = []
        for combo in combinations:
            merged = {node.id: combined_label_values}
            merged.update(combo)
            results.append(merged)
        return results

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _resolve_selected_children(self, group_node: FMNode, selected_raw: List[str]) -> List[FMNode]:
        if not selected_raw:
            return []

        selected: List[FMNode] = []
        seen = set()
        child_by_id = {child.id: child for child in group_node.children}
        child_by_name = {child.name: child for child in group_node.children}

        for raw in selected_raw:
            child = child_by_id.get(raw) or child_by_name.get(raw)
            if child and child.id not in seen:
                selected.append(child)
                seen.add(child.id)

        return selected

    def _is_child_active(self, child: FMNode, ctx: Dict[str, Any]) -> bool:
        if child.mandatory:
            return True
        if child.id in ctx["selected_features"]:
            return True
        if child.id in ctx["selected_options"] and ctx["selected_options"][child.id]:
            return True
        if child.id in ctx["string_values"] and self._get_string_values(child.id, ctx["string_values"]):
            return True
        return self._has_descendant_selection(child.id, ctx)

    def _has_descendant_selection(self, node_id: str, ctx: Dict[str, Any]) -> bool:
        prefix = f"{node_id}."
        for key, values in ctx["selected_options"].items():
            if key.startswith(prefix) and values:
                return True
        for key, values in ctx["string_values"].items():
            if key.startswith(prefix) and self._get_string_values(key, ctx["string_values"]):
                return True
        for feature_id in ctx["selected_features"]:
            if str(feature_id).startswith(prefix):
                return True
        return False

    def _is_group_explicitly_selected(self, node: FMNode, ctx: Dict[str, Any]) -> bool:
        """Return True when a group is directly selected via its parent selection list."""
        for selected_values in ctx["selected_options"].values():
            if not isinstance(selected_values, list):
                continue
            for raw in selected_values:
                if raw == node.id or raw == node.name:
                    return True
        return False

    def _get_string_values(self, node_id: str, string_values: Dict[str, Any]) -> List[str]:
        raw = string_values.get(node_id, [])
        if isinstance(raw, str):
            values = [value.strip() for value in raw.split(",") if value.strip()]
            return values
        if isinstance(raw, list):
            values = [str(value).strip() for value in raw if str(value).strip()]
            return values
        return []

    def _merge_variant_lists(
        self,
        left_variants: List[Dict[str, Any]],
        right_variants: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not right_variants:
            return []
        merged: List[Dict[str, Any]] = []
        for left in left_variants:
            for right in right_variants:
                combined = dict(left)
                combined.update(right)
                merged.append(combined)
        return merged

    def _product_variants(self, variant_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not variant_lists:
            return [{}]

        product: List[Dict[str, Any]] = [{}]
        for variants in variant_lists:
            product = self._merge_variant_lists(product, variants)
        return product

    def _clean_variant(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in variant.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, list) and not value:
                continue
            cleaned[key] = value
        return cleaned

    def _build_constraints(self, variant: Dict[str, Any]) -> List[Dict[str, Any]]:
        constraints: List[Dict[str, Any]] = []
        for key, value in variant.items():
            node = self._node_index.get(key)
            label = node.name if node else key
            constraints.append(
                {
                    "id": key,
                    "label": label,
                    "value": value,
                }
            )
        return constraints

    # ── Constraint evaluation ───────────────────────────────────────────────
    def _satisfies_constraints(self, variant: Dict[str, Any]) -> bool:
        if not self._constraints:
            return True
        active = self._collect_active_features(variant)
        return all(self._evaluate(c, active) for c in self._constraints)

    def _evaluate(self, constraint: FMConstraint, active: set) -> bool:
        op = constraint.operator

        if op == "var":
            return constraint.variable in active

        if op == "not":
            return not self._evaluate(constraint.operands[0], active)

        if op == "and":
            return all(self._evaluate(o, active) for o in constraint.operands)

        if op == "or":
            return any(self._evaluate(o, active) for o in constraint.operands)

        if op == "imp":
            antecedent = self._evaluate(constraint.operands[0], active)
            consequent = self._evaluate(constraint.operands[1], active)
            return (not antecedent) or consequent

        if op == "eq":
            values = [self._evaluate(o, active) for o in constraint.operands]
            return len(set(values)) <= 1

        return True

    def _collect_active_features(self, variant: Dict[str, Any]) -> set:
        """Collect all feature names active in a variant."""
        active: set = set()
        for key, value in variant.items():
            if key.startswith("__"):
                continue
            node = self._node_index.get(key)
            if node:
                active.add(node.name)
            if isinstance(value, str):
                val_node = self._node_index.get(f"{key}.{value}")
                if val_node:
                    active.add(val_node.name)
                else:
                    active.add(value)
            elif isinstance(value, list):
                for v in value:
                    v_str = str(v)
                    val_node = self._node_index.get(f"{key}.{v_str}")
                    if val_node:
                        active.add(val_node.name)
                    else:
                        active.add(v_str)
            elif isinstance(value, bool) and value:
                pass
        return active
