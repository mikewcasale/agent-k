"""SAGE docstring extractor and validator.

@notice: |
    Extracts SAGE tags from docstrings and emits artifacts for the SAGE VS Code extension.

@dev: |
    Parses Python source with AST, ignores fenced code blocks, and validates
    tag grammar (no hybrid scalar+block). Emits JSON artifacts in .sage/ including:
    - index.json: Full component index with metadata
    - components.json: CRG component graph (legacy, kept for compatibility)
    - canonical-homes.json: Canonical home mappings
    - graph.json: CRG visualization data
    - agent-guidance.json: Aggregated @agent-guidance tags
    - patterns.json: Pattern catalog from @pattern tags
    - anti-patterns.json: Anti-pattern warnings
    - errors.json: Validation errors
    - components/: Per-component JSON payloads (spec-aligned)
    - similar/: Per-component similar reference summaries

@graph:
    id: agent_k.tools.sageextract
    provides:
        - agent_k.tools.sageextract:run
        - agent_k.tools.sageextract:main
    pattern: tooling

@agent-guidance:
    do:
        - "Use this script to validate SAGE docstrings before release."
    do_not:
        - "Hand-edit .sage artifacts; regenerate instead."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import argparse
import ast
import json
import re
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, TypeAliasType

__all__ = ("main", "run")

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PACKAGE_ROOT: Final[Path] = ROOT / "agent_k"
DEFAULT_OUT_DIR: Final[Path] = ROOT / ".sage"

ComponentType = Literal["module", "class", "function", "method"]

TagValue = TypeAliasType("TagValue", dict[str, Any] | list[Any] | str | int | float | bool | None)


@dataclass(frozen=True)
class DocBlock:
    """Docstring payload with source metadata."""

    text: str
    source: str
    start_line: int
    component_id: str
    component_type: ComponentType
    module: str
    qualname: str | None


@dataclass(frozen=True)
class ModuleInfo:
    """Module metadata for validation."""

    module: str
    source: str
    exported: set[str]
    public_candidates: set[str]
    has_defs: bool


@dataclass(frozen=True)
class ParsedDoc:
    """Parsed component docstring data."""

    component_id: str
    component_type: ComponentType
    module: str
    qualname: str | None
    source: str
    start_line: int
    summary: str | None
    tags: dict[str, TagValue]
    tag_lines: dict[str, int]


@dataclass(frozen=True)
class LineToken:
    """Docstring line token."""

    text: str
    line_no: int


@dataclass(frozen=True)
class YamlLine:
    """Normalized YAML-like line."""

    indent: int
    text: str
    line_no: int


def run(out_dir: Path, *, validate: bool, emit: bool) -> int:
    """Run SAGE extraction and validation.

    @notice: |
        Main entry point for SAGE artifact generation and validation.

    @dev: |
        Collects docblocks from all Python files, extracts tags, validates grammar,
        and emits JSON artifacts for the VS Code extension.
    """
    docblocks, modules = _collect_docblocks(PACKAGE_ROOT)
    parsed_docs, parse_errors = _extract_docstrings(docblocks)

    validation_errors = _validate_components(parsed_docs, modules)
    errors = [*parse_errors, *validation_errors]

    has_fatal = _has_fatal_errors(errors)

    if validate and errors:
        _write_json(out_dir / "errors.json", errors if emit else errors)
    if emit:
        out_dir.mkdir(parents=True, exist_ok=True)

        components = _build_components(parsed_docs)
        canonical_homes = _build_canonical_homes(parsed_docs)
        _write_json(out_dir / "components.json", components)
        _write_json(out_dir / "canonical-homes.json", canonical_homes)

        index = _build_index(parsed_docs, components)
        graph = _build_graph(parsed_docs, components)
        agent_guidance = _build_agent_guidance(parsed_docs)
        patterns = _build_patterns(parsed_docs)
        anti_patterns = _build_anti_patterns(parsed_docs)

        _write_json(out_dir / "index.json", index)
        _write_json(out_dir / "graph.json", graph)
        _write_json(out_dir / "agent-guidance.json", agent_guidance)
        _write_json(out_dir / "patterns.json", patterns)
        _write_json(out_dir / "anti-patterns.json", anti_patterns)
        _write_json(out_dir / "errors.json", errors)

        _write_component_files(out_dir / "components", parsed_docs)
        _write_similar_files(out_dir / "similar", parsed_docs)

    return 1 if has_fatal else 0


def main() -> None:
    """CLI entrypoint for SAGE extraction."""
    parser = argparse.ArgumentParser(description="Extract SAGE docstring metadata.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for artifacts.")
    parser.add_argument("--emit", action="store_true", help="Emit .sage artifacts.")
    parser.add_argument("--validate", action="store_true", help="Validate tag grammar.")
    args = parser.parse_args()

    emit = args.emit or not args.validate
    exit_code = run(args.out_dir, validate=args.validate, emit=emit)
    raise SystemExit(exit_code)


def _collect_docblocks(root: Path) -> tuple[list[DocBlock], list[ModuleInfo]]:
    docblocks: list[DocBlock] = []
    modules: list[ModuleInfo] = []
    for path in sorted(root.rglob("*.py")):
        module_blocks, module_info = _extract_docblocks(path)
        docblocks.extend(module_blocks)
        modules.append(module_info)
    return docblocks, modules


def _extract_docblocks(path: Path) -> tuple[list[DocBlock], ModuleInfo]:
    text = path.read_text(encoding="utf-8")
    module = ast.parse(text)
    source = str(path.relative_to(ROOT))
    module_name = _module_name(path)

    docblocks: list[DocBlock] = []
    exported = _extract_all_exports(module)
    public_candidates = _extract_public_candidates(module)
    has_defs = bool(public_candidates)

    module_doc = _docblock_for(module, source, module_name, "module", None)
    if module_doc is not None:
        docblocks.append(module_doc)

    def walk(node: ast.AST, parents: list[str], *, in_class: bool) -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qualname = ".".join([*parents, child.name])
                docblock = _docblock_for(child, source, module_name, "class", qualname)
                if docblock is not None:
                    docblocks.append(docblock)
                walk(child, [*parents, child.name], in_class=True)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join([*parents, child.name])
                kind: ComponentType = "method" if in_class else "function"
                docblock = _docblock_for(child, source, module_name, kind, qualname)
                if docblock is not None:
                    docblocks.append(docblock)
                walk(child, [*parents, child.name], in_class=in_class)

    walk(module, [], in_class=False)

    module_info = ModuleInfo(
        module=module_name, source=source, exported=exported, public_candidates=public_candidates, has_defs=has_defs
    )

    return docblocks, module_info


def _module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["agent_k", *parts]) if parts else "agent_k"


def _docblock_for(
    node: ast.AST, source: str, module: str, component_type: ComponentType, qualname: str | None
) -> DocBlock | None:
    if not hasattr(node, "body"):
        return None
    body = getattr(node, "body")
    if not body:
        return None
    first = body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)):
        return None
    if not isinstance(first.value.value, str):
        return None
    text = first.value.value
    start_line = first.lineno
    component_id = module if component_type == "module" else f"{module}:{qualname}"
    return DocBlock(
        text=text,
        source=source,
        start_line=start_line,
        component_id=component_id,
        component_type=component_type,
        module=module,
        qualname=qualname,
    )


def _extract_all_exports(module: ast.Module) -> set[str]:
    exported: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = node.value if isinstance(node, ast.Assign) else node.value
                    exported.update(_extract_str_iterable(value))
    return exported


def _extract_public_candidates(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


def _extract_str_iterable(value: ast.AST | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (ast.List, ast.Tuple)):
        items: set[str] = set()
        for elt in value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                items.add(elt.value)
        return items
    return set()


def _extract_docstrings(docblocks: Iterable[DocBlock]) -> tuple[list[ParsedDoc], list[dict[str, Any]]]:
    parsed: list[ParsedDoc] = []
    errors: list[dict[str, Any]] = []

    for block in docblocks:
        tags, tag_lines, tag_errors = _parse_docstring(block)
        errors.extend(tag_errors)
        summary = _summary(block.text)
        parsed.append(
            ParsedDoc(
                component_id=block.component_id,
                component_type=block.component_type,
                module=block.module,
                qualname=block.qualname,
                source=block.source,
                start_line=block.start_line,
                summary=summary,
                tags=tags,
                tag_lines=tag_lines,
            )
        )

    return parsed, errors


def _summary(text: str) -> str | None:
    for line in textwrap.dedent(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("@"):
            continue
        return stripped
    return None


def _parse_docstring(block: DocBlock) -> tuple[dict[str, TagValue], dict[str, int], list[dict[str, Any]]]:
    tags: dict[str, TagValue] = {}
    tag_lines: dict[str, int] = {}
    tag_types: dict[str, str] = {}
    errors: list[dict[str, Any]] = []

    lines = textwrap.dedent(block.text).splitlines()
    in_fence = False
    current_tag: str | None = None
    current_style: str | None = None
    current_start: int | None = None
    header_indent = 0
    buffer: list[LineToken] = []

    def flush() -> None:
        nonlocal current_tag, current_style, current_start, buffer, header_indent
        if current_tag is None:
            return
        line_no = current_start or block.start_line
        tag_name = current_tag
        raw_value: TagValue | None = None
        tag_error: str | None = None

        if current_style == "scalar":
            if len(buffer) > 1 and any(token.text.strip() for token in buffer[1:]):
                tag_error = f"Hybrid tag value for @{tag_name}"
            raw_value = _parse_scalar(buffer[0].text.strip()) if buffer else ""
        else:
            raw_value, tag_error = _parse_block_tag(
                buffer, base_indent=header_indent + 4, style=current_style, source=block.source
            )

        if tag_error:
            errors.append(_error(block, line_no, tag_error))
        if raw_value is None:
            raw_value = ""

        _merge_tag(tags, tag_types, tag_lines, tag_name, raw_value, line_no, errors, block)

        current_tag = None
        current_style = None
        current_start = None
        header_indent = 0
        buffer = []

    for idx, line in enumerate(lines):
        line_no = block.start_line + idx
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("@"):
            flush()
            header_indent = len(line) - len(stripped)
            tag_name, style, inline_value, error = _parse_tag_header(stripped)
            if error:
                errors.append(_error(block, line_no, error))
                current_tag = None
                current_style = None
                current_start = None
                header_indent = 0
                buffer = []
                continue
            current_tag = tag_name
            current_style = style
            current_start = line_no
            if inline_value:
                buffer.append(LineToken(text=inline_value, line_no=line_no))
            continue
        if current_tag is not None:
            base_indent = header_indent + 4
            if current_style == "scalar":
                indent = len(line) - len(stripped)
                if line.strip() and indent >= base_indent:
                    buffer.append(LineToken(text=line, line_no=line_no))
                else:
                    flush()
                continue
            if not line.strip():
                buffer.append(LineToken(text=line, line_no=line_no))
                continue
            indent = len(line) - len(stripped)
            if indent < base_indent:
                flush()
                continue
            buffer.append(LineToken(text=line, line_no=line_no))

    flush()
    return tags, tag_lines, errors


def _parse_tag_header(line: str) -> tuple[str, str | None, str | None, str | None]:
    if ":" not in line:
        return "", None, None, "Tag header missing ':'"

    if line.startswith("@custom:"):
        remainder = line[1:]
        idx = remainder.find(":", len("custom:"))
        if idx == -1:
            return "", None, None, "Custom tag missing name separator ':'"
        tag_name = remainder[:idx].strip()
        rest = remainder[idx + 1 :]
    else:
        tag_name, rest = line[1:].split(":", 1)
        tag_name = tag_name.strip()

    if not tag_name:
        return "", None, None, "Empty tag name"

    rest_stripped = rest.strip()
    if rest_stripped == "":
        return tag_name, None, None, None
    if rest_stripped in {"|", ">"}:
        return tag_name, rest_stripped, None, None
    return tag_name, "scalar", rest_stripped, None


def _parse_block_tag(
    buffer: list[LineToken], *, base_indent: int, style: str | None, source: str
) -> tuple[TagValue | None, str | None]:
    if style in {"|", ">"}:
        text, error = _parse_block_scalar(buffer, base_indent=base_indent, style=style, source=source)
        return text, error

    yaml_lines, errors = _normalize_yaml_lines(buffer, base_indent=base_indent, source=source)
    if errors:
        return None, errors[0]["message"]
    if not yaml_lines:
        return {}, None
    value, _, parse_error = _parse_yaml_value(yaml_lines, 0, indent=0)
    return value, parse_error


def _parse_block_scalar(
    buffer: list[LineToken], *, base_indent: int, style: str, source: str
) -> tuple[str, str | None]:
    lines: list[str] = []
    error: str | None = None

    for token in buffer:
        if not token.text.strip():
            lines.append("")
            continue
        if "\t" in token.text:
            error = "Tabs are not allowed in block scalars"
            continue
        indent = len(token.text) - len(token.text.lstrip(" "))
        if indent < base_indent:
            error = "Block scalar indentation must be at least 4 spaces"
            continue
        lines.append(token.text[base_indent:])

    text = "\n".join(lines)
    if style == ">":
        text = _fold_block(text)
    return text.rstrip(), error


def _fold_block(text: str) -> str:
    folded: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if buffer:
                folded.append(" ".join(buffer))
                buffer = []
            folded.append("")
            continue
        buffer.append(line)
    if buffer:
        folded.append(" ".join(buffer))
    return "\n".join(folded)


def _normalize_yaml_lines(
    buffer: list[LineToken], *, base_indent: int, source: str
) -> tuple[list[YamlLine], list[dict[str, Any]]]:
    lines: list[YamlLine] = []
    errors: list[dict[str, Any]] = []
    for token in buffer:
        if not token.text.strip():
            lines.append(YamlLine(indent=0, text="", line_no=token.line_no))
            continue
        if "\t" in token.text:
            errors.append(_error_raw(source, token.line_no, "Tabs are not allowed in tags"))
            continue
        indent = len(token.text) - len(token.text.lstrip(" "))
        if indent < base_indent:
            errors.append(_error_raw(source, token.line_no, "Tag block indentation must be at least 4 spaces"))
            continue
        rel_indent = indent - base_indent
        if rel_indent % 4 != 0:
            errors.append(_error_raw(source, token.line_no, "Tag block indentation must use 4 spaces"))
        lines.append(YamlLine(indent=rel_indent, text=token.text[indent:], line_no=token.line_no))
    return lines, errors


def _parse_yaml_value(lines: list[YamlLine], start: int, *, indent: int) -> tuple[TagValue, int, str | None]:
    index = _skip_blanks(lines, start)
    if index >= len(lines) or lines[index].indent < indent:
        return {}, index, None
    if lines[index].text.startswith("- "):
        return _parse_list(lines, index, indent)
    if _is_map_line(lines[index].text):
        return _parse_map(lines, index, indent)

    texts: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.text == "":
            index += 1
            continue
        if line.indent < indent:
            break
        if line.indent > indent:
            return "\n".join(texts).strip(), index, f"Unexpected indentation at line {line.line_no}"
        texts.append(line.text)
        index += 1
    return "\n".join(texts).strip(), index, None


def _parse_list(lines: list[YamlLine], start: int, indent: int) -> tuple[list[Any], int, str | None]:
    items: list[Any] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if line.text == "":
            index += 1
            continue
        if line.indent < indent:
            break
        if line.indent > indent:
            return items, index, f"Unexpected indentation at line {line.line_no}"
        if not line.text.startswith("- "):
            return items, index, f"Expected list item at line {line.line_no}"
        item_text = line.text[2:].strip()
        if item_text in {"|", ">"}:
            scalar_val, next_index, error = _parse_block_scalar_yaml(lines, index + 1, indent + 4, item_text)
            if error:
                return items, next_index, error
            items.append(scalar_val)
            index = next_index
            continue
        if item_text == "":
            nested_val, next_index, error = _parse_yaml_value(lines, index + 1, indent=indent + 4)
            if error:
                return items, next_index, error
            items.append(nested_val)
            index = next_index
            continue
        if _is_map_line(item_text):
            map_val, next_index, error = _parse_inline_map_item(lines, index, indent)
            if error:
                return items, next_index, error
            items.append(map_val)
            index = next_index
            continue
        items.append(_parse_scalar(item_text))
        index += 1
    return items, index, None


def _parse_inline_map_item(lines: list[YamlLine], index: int, indent: int) -> tuple[dict[str, Any], int, str | None]:
    line = lines[index]
    item_text = line.text[2:].strip()
    key, value_text = _split_map_entry(item_text)
    if key is None:
        return {}, index + 1, f"Invalid map entry at line {line.line_no}"
    data: dict[str, Any] = {}
    normalized_key = _normalize_key(key)
    next_index = index + 1

    if value_text in {"|", ">"}:
        scalar_val, next_index, error = _parse_block_scalar_yaml(lines, next_index, indent + 4, value_text)
        if error:
            return data, next_index, error
        data[normalized_key] = scalar_val
    elif value_text == "":
        nested_val, next_index, error = _parse_yaml_value(lines, next_index, indent=indent + 4)
        if error:
            return data, next_index, error
        data[normalized_key] = nested_val
    else:
        data[normalized_key] = _parse_scalar(value_text)

    while next_index < len(lines):
        next_line = lines[next_index]
        if next_line.text == "":
            next_index += 1
            continue
        if next_line.indent < indent + 4:
            break
        if next_line.indent > indent + 4:
            return data, next_index, f"Unexpected indentation at line {next_line.line_no}"
        if next_line.text.startswith("- "):
            break
        key, value_text = _split_map_entry(next_line.text)
        if key is None:
            return data, next_index, f"Invalid map entry at line {next_line.line_no}"
        normalized_key = _normalize_key(key)
        next_index += 1
        if value_text in {"|", ">"}:
            inner_scalar, next_index, error = _parse_block_scalar_yaml(lines, next_index, indent + 8, value_text)
            if error:
                return data, next_index, error
            data[normalized_key] = inner_scalar
            continue
        if value_text == "":
            inner_nested, next_index, error = _parse_yaml_value(lines, next_index, indent=indent + 8)
            if error:
                return data, next_index, error
            data[normalized_key] = inner_nested
            continue
        data[normalized_key] = _parse_scalar(value_text)
    return data, next_index, None


def _parse_map(lines: list[YamlLine], start: int, indent: int) -> tuple[dict[str, Any], int, str | None]:
    data: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line = lines[index]
        if line.text == "":
            index += 1
            continue
        if line.indent < indent:
            break
        if line.indent > indent:
            return data, index, f"Unexpected indentation at line {line.line_no}"
        if line.text.startswith("- "):
            return data, index, f"Unexpected list item at line {line.line_no}"
        key, value_text = _split_map_entry(line.text)
        if key is None:
            return data, index, f"Invalid map entry at line {line.line_no}"
        normalized_key = _normalize_key(key)
        index += 1
        if value_text in {"|", ">"}:
            scalar_val, index, error = _parse_block_scalar_yaml(lines, index, indent + 4, value_text)
            if error:
                return data, index, error
            data[normalized_key] = scalar_val
            continue
        if value_text == "":
            nested_val, index, error = _parse_yaml_value(lines, index, indent=indent + 4)
            if error:
                return data, index, error
            data[normalized_key] = nested_val
            continue
        data[normalized_key] = _parse_scalar(value_text)
    return data, index, None


def _parse_block_scalar_yaml(lines: list[YamlLine], start: int, indent: int, style: str) -> tuple[str, int, str | None]:
    buffer: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if line.text == "":
            buffer.append("")
            index += 1
            continue
        if line.indent < indent:
            break
        buffer.append((" " * (line.indent - indent)) + line.text)
        index += 1
    text = "\n".join(buffer)
    if style == ">":
        text = _fold_block(text)
    return text.rstrip(), index, None


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if stripped in {"null", "none"}:
        return None
    if _is_int(stripped):
        return int(stripped)
    if _is_float(stripped):
        return float(stripped)
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped


def _is_int(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", value))


def _is_float(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+\.\d+", value))


def _is_map_line(text: str) -> bool:
    return _split_map_entry(text)[0] is not None


def _split_map_entry(text: str) -> tuple[str | None, str]:
    if ":" not in text:
        return None, ""
    key, rest = text.split(":", 1)
    if not key.strip():
        return None, ""
    if rest == "":
        return key.strip(), ""
    if rest.startswith(" "):
        return key.strip(), rest.strip()
    return None, ""


def _skip_blanks(lines: list[YamlLine], start: int) -> int:
    index = start
    while index < len(lines) and lines[index].text == "":
        index += 1
    return index


def _merge_tag(
    tags: dict[str, TagValue],
    tag_types: dict[str, str],
    tag_lines: dict[str, int],
    raw_tag: str,
    value: TagValue,
    line_no: int,
    errors: list[dict[str, Any]],
    block: DocBlock,
) -> None:
    if raw_tag.startswith("custom:"):
        custom_key = _normalize_key(raw_tag.split(":", 1)[1])
        raw_tag = "custom"
        value = {custom_key: value}

    tag_name = _normalize_key(raw_tag)
    tag_type = _tag_type(value)

    if tag_name not in tags:
        tags[tag_name] = value
        tag_types[tag_name] = tag_type
        tag_lines[tag_name] = line_no
        return

    existing_type = tag_types[tag_name]
    if existing_type != tag_type:
        errors.append(
            _error(block, line_no, f"Duplicate tag @{tag_name} with mismatched types ({existing_type} vs {tag_type})")
        )
        return

    if tag_type == "map":
        existing_map = tags[tag_name]
        assert isinstance(existing_map, dict)
        assert isinstance(value, dict)
        tags[tag_name] = {**existing_map, **value}
        return
    if tag_type == "list":
        existing_list = tags[tag_name]
        assert isinstance(existing_list, list)
        assert isinstance(value, list)
        tags[tag_name] = [*existing_list, *value]
        return

    tags[tag_name] = value


def _tag_type(value: TagValue) -> str:
    if isinstance(value, dict):
        return "map"
    if isinstance(value, list):
        return "list"
    return "scalar"


def _normalize_key(key: str) -> str:
    return ".".join(segment.replace("-", "_") for segment in key.split("."))


def _build_components(parsed_docs: Iterable[ParsedDoc]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for doc in parsed_docs:
        graph = _as_dict(doc.tags.get("graph"))
        if not graph:
            continue
        component_id = graph.get("id") or doc.component_id
        components.append(
            {
                "id": component_id,
                "provides": _ensure_list(graph.get("provides")),
                "consumes": _ensure_list(graph.get("consumes")),
                "pattern": graph.get("pattern"),
                "source": doc.source,
                "line": doc.start_line,
            }
        )
    return components


def _build_canonical_homes(parsed_docs: Iterable[ParsedDoc]) -> list[dict[str, Any]]:
    homes: list[dict[str, Any]] = []
    for doc in parsed_docs:
        canonical = doc.tags.get("canonical_home")
        if canonical is None:
            continue
        parsed = _as_dict(canonical)
        homes.append(
            {
                "for": _ensure_list(parsed.get("for")),
                "notes": parsed.get("notes"),
                "source": doc.source,
                "line": doc.start_line,
            }
        )
    return homes


def _build_index(parsed_docs: Iterable[ParsedDoc], components: list[dict[str, Any]]) -> dict[str, Any]:
    tags_by_source: dict[str, list[dict[str, Any]]] = {}
    tag_count = 0

    for doc in parsed_docs:
        entry_list = tags_by_source.setdefault(doc.source, [])
        for name, value in doc.tags.items():
            tag_count += 1
            entry_list.append(
                {
                    "component": doc.component_id,
                    "name": name,
                    "value": _compact_tag_value(value),
                    "line": doc.tag_lines.get(name, doc.start_line),
                }
            )

    return {
        "version": "1.3.6",
        "generated": _iso_timestamp(),
        "component_count": len(components),
        "tag_count": tag_count,
        "sources": list(tags_by_source.keys()),
        "components": [c for c in components if c.get("id")],
        "tags_by_source": tags_by_source,
    }


def _build_graph(parsed_docs: Iterable[ParsedDoc], components: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    for comp in components:
        comp_id = comp.get("id")
        if not comp_id or comp_id in seen_nodes:
            continue
        seen_nodes.add(comp_id)
        nodes.append(
            {
                "id": comp_id,
                "label": comp_id.split(":")[-1].split(".")[-1],
                "source": comp.get("source"),
                "pattern": comp.get("pattern"),
            }
        )

    for doc in parsed_docs:
        graph = _as_dict(doc.tags.get("graph"))
        comp_id = (graph.get("id") if graph else None) or doc.component_id
        for key in ("provides", "consumes", "calls", "extends", "implements", "replaces"):
            targets = _ensure_list(graph.get(key)) if graph else []
            for target in targets:
                edges.append({"source": comp_id, "target": target, "type": key})
        similar = doc.tags.get("similar")
        for entry in _ensure_list(similar):
            if isinstance(entry, dict):
                target = entry.get("id")
            else:
                target = entry
            if target:
                edges.append({"source": comp_id, "target": target, "type": "similar"})

    return {"version": "1.3.6", "generated": _iso_timestamp(), "nodes": nodes, "edges": edges}


def _build_agent_guidance(parsed_docs: Iterable[ParsedDoc]) -> dict[str, Any]:
    guidance_entries: list[dict[str, Any]] = []

    for doc in parsed_docs:
        guidance = _as_dict(doc.tags.get("agent_guidance"))
        if not guidance:
            continue
        guidance_entries.append(
            {
                "component": doc.component_id,
                "source": doc.source,
                "line": doc.start_line,
                "do": _ensure_list(guidance.get("do")),
                "do_not": _ensure_list(guidance.get("do_not")),
            }
        )

    return {
        "version": "1.3.6",
        "generated": _iso_timestamp(),
        "count": len(guidance_entries),
        "entries": guidance_entries,
    }


def _build_patterns(parsed_docs: Iterable[ParsedDoc]) -> dict[str, Any]:
    patterns: dict[str, list[dict[str, Any]]] = {}

    for doc in parsed_docs:
        pattern = doc.tags.get("pattern")
        if pattern is None:
            continue
        if isinstance(pattern, dict):
            pattern_name = pattern.get("name") or pattern.get("pattern") or "unknown"
            entry = {
                "component": doc.component_id,
                "source": doc.source,
                "line": doc.start_line,
                "rationale": pattern.get("rationale"),
                "violations": _ensure_list(pattern.get("violations")),
            }
        else:
            pattern_name = str(pattern).strip() if pattern else "unknown"
            entry = {"component": doc.component_id, "source": doc.source, "line": doc.start_line}
        patterns.setdefault(pattern_name, []).append(entry)

    return {"version": "1.3.6", "generated": _iso_timestamp(), "pattern_count": len(patterns), "patterns": patterns}


def _build_anti_patterns(parsed_docs: Iterable[ParsedDoc]) -> dict[str, Any]:
    anti_patterns: list[dict[str, Any]] = []

    for doc in parsed_docs:
        anti = doc.tags.get("anti_patterns")
        if anti is None:
            continue
        if isinstance(anti, dict):
            entry = {
                "component": doc.component_id,
                "source": doc.source,
                "line": doc.start_line,
                "items": _ensure_list(anti.get("items")),
                "alternatives": _ensure_list(anti.get("alternatives")),
            }
        else:
            entry = {
                "component": doc.component_id,
                "source": doc.source,
                "line": doc.start_line,
                "items": _ensure_list(anti),
                "alternatives": [],
            }
        anti_patterns.append(entry)

    return {"version": "1.3.6", "generated": _iso_timestamp(), "count": len(anti_patterns), "entries": anti_patterns}


def _write_component_files(out_dir: Path, parsed_docs: Iterable[ParsedDoc]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in parsed_docs:
        payload = _component_payload(doc)
        file_name = f"{doc.component_id.replace(':', '.')}.json"
        _write_json(out_dir / file_name, payload)


def _write_similar_files(out_dir: Path, parsed_docs: Iterable[ParsedDoc]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in parsed_docs:
        similar = _ensure_list(doc.tags.get("similar"))
        if not similar:
            continue
        lines = [f"# Similar components for {doc.component_id}", ""]
        for entry in similar:
            if isinstance(entry, dict):
                target = entry.get("id")
                reason = entry.get("when")
                if target and reason:
                    lines.append(f"- {target}: {reason}")
                elif target:
                    lines.append(f"- {target}")
            else:
                lines.append(f"- {entry}")
        file_name = f"{doc.component_id.replace(':', '.')}.md"
        out_dir.joinpath(file_name).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _component_payload(doc: ParsedDoc) -> dict[str, Any]:
    payload = {
        "id": doc.component_id,
        "type": doc.component_type,
        "summary": doc.summary,
        "notice": doc.tags.get("notice"),
        "dev": doc.tags.get("dev"),
        "graph": doc.tags.get("graph"),
        "pattern": doc.tags.get("pattern"),
        "similar": doc.tags.get("similar"),
        "agent_guidance": doc.tags.get("agent_guidance"),
        "human_review": doc.tags.get("human_review"),
        "anti_patterns": doc.tags.get("anti_patterns"),
        "canonical_home": doc.tags.get("canonical_home"),
        "custom": doc.tags.get("custom"),
        "source": doc.source,
        "line": doc.start_line,
    }
    return payload


def _validate_components(parsed_docs: list[ParsedDoc], modules: list[ModuleInfo]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    module_lookup = {module.module: module for module in modules}
    component_ids = {doc.component_id for doc in parsed_docs}
    for doc in parsed_docs:
        graph_id = _as_dict(doc.tags.get("graph")).get("id")
        if graph_id:
            component_ids.add(str(graph_id))
        graph = _as_dict(doc.tags.get("graph"))
        for provided in _ensure_list(graph.get("provides")):
            component_ids.add(str(provided))

    for doc in parsed_docs:
        if doc.component_type == "module":
            module = module_lookup.get(doc.component_id)
            if module and module.has_defs and "graph" not in doc.tags:
                errors.append(_validation_error(doc, "V001", "ERROR", "Module missing @graph"))
            if "graph" in doc.tags:
                graph = _as_dict(doc.tags.get("graph"))
                if not graph.get("id"):
                    errors.append(_validation_error(doc, "V001", "ERROR", "@graph missing id"))
        if doc.component_type == "class":
            if "pattern" not in doc.tags:
                errors.append(_validation_error(doc, "V002", "ERROR", "Class missing @pattern"))
            if _requires_collaborators(doc) and "collaborators" not in doc.tags:
                errors.append(_validation_error(doc, "V003", "ERROR", "Class missing @collaborators"))
        if doc.component_type in {"module", "class", "function"}:
            if _is_public_component(doc, module_lookup):
                if "notice" not in doc.tags or "dev" not in doc.tags:
                    errors.append(_validation_error(doc, "V005", "WARN", "Public component missing @notice or @dev"))

        _validate_similar_refs(doc, component_ids, errors)
        _validate_semantics(doc, errors)

    return errors


def _requires_collaborators(doc: ParsedDoc) -> bool:
    if not doc.qualname:
        return False
    name = doc.qualname.split(".")[-1]
    summary = doc.summary.lower() if doc.summary else ""
    return name.endswith("Deps") or "dependencies" in summary or "dependency" in summary


def _is_public_component(doc: ParsedDoc, module_lookup: dict[str, ModuleInfo]) -> bool:
    if doc.component_type == "module":
        return True
    if not doc.qualname:
        return False
    name = doc.qualname.split(".")[-1]
    if name.startswith("_"):
        return False
    module = module_lookup.get(doc.module)
    if not module:
        return True
    exports = module.exported or module.public_candidates
    return name in exports


def _validate_similar_refs(doc: ParsedDoc, component_ids: set[str], errors: list[dict[str, Any]]) -> None:
    refs: list[str] = []
    graph = _as_dict(doc.tags.get("graph"))
    for key in ("provides", "consumes", "calls", "extends", "implements", "replaces"):
        refs.extend([str(ref) for ref in _ensure_list(graph.get(key))])
    for entry in _ensure_list(doc.tags.get("similar")):
        if isinstance(entry, dict):
            target = entry.get("id")
            if target:
                refs.append(str(target))
        elif entry:
            refs.append(str(entry))
    for ref in refs:
        if ref.startswith("agent_k") and ref not in component_ids:
            errors.append(_validation_error(doc, "C003", "WARN", f"Unknown component reference: {ref}"))


def _validate_semantics(doc: ParsedDoc, errors: list[dict[str, Any]]) -> None:
    pure = _as_dict(doc.tags.get("pure"))
    total = _as_dict(doc.tags.get("total"))
    effects = _as_dict(doc.tags.get("effects"))
    errors_tag = _as_dict(doc.tags.get("errors"))

    pure_value = _as_bool(pure.get("value")) if pure else None
    total_value = _as_bool(total.get("value")) if total else None
    if pure_value is True:
        if effects and (_ensure_list(effects.get("io")) or _ensure_list(effects.get("external"))):
            errors.append(_validation_error(doc, "S001", "WARN", "@pure true with external effects"))
    if pure_value is False and not effects:
        errors.append(_validation_error(doc, "C004", "WARN", "@pure false without @effects"))
    if total_value is True and errors_tag and _ensure_list(errors_tag.get("terminal")):
        errors.append(_validation_error(doc, "S002", "INFO", "@total true with terminal errors"))


def _as_dict(value: TagValue | None) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _compact_tag_value(value: TagValue) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=True)
    else:
        text = str(value)
    return text[:200] if len(text) > 200 else text


def _iso_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _validation_error(doc: ParsedDoc, code: str, level: str, message: str) -> dict[str, Any]:
    return {
        "source": doc.source,
        "line": doc.start_line,
        "message": message,
        "code": code,
        "level": level,
        "component": doc.component_id,
    }


def _error(block: DocBlock, line_no: int, message: str) -> dict[str, Any]:
    return {"source": block.source, "line": line_no, "message": message}


def _error_raw(source: str, line_no: int, message: str) -> dict[str, Any]:
    return {"source": source, "line": line_no, "message": message}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _has_fatal_errors(errors: Iterable[dict[str, Any]]) -> bool:
    for err in errors:
        level = err.get("level")
        if level is None:
            return True
        if str(level).upper() == "ERROR":
            return True
    return False


if __name__ == "__main__":
    main()
