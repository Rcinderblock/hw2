"""Strict JSON decoding and safe file handling for console commands."""

import json
import tempfile
from pathlib import Path


def unique_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON содержит повторяющиеся ключи")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise ValueError("NaN и Infinity не входят в формат JSON")


def decode_json(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=unique_json_keys,
        parse_constant=reject_json_constant,
    )


def load_sample_inputs(directory: Path) -> list[tuple[str, str]]:
    paths = sorted(directory.glob("*.txt"))
    if not paths:
        raise ValueError(f"Нет входных текстов в {directory}")
    return [(path.stem, path.read_text(encoding="utf-8")) for path in paths]


def validate_output_path(
    output: Path | None, source_paths: list[Path]
) -> None:
    if output and any(
        output.resolve() == path.resolve() for path in source_paths
    ):
        raise ValueError("Файл результата не должен заменять входной файл")


def write_json_output(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        # Replace the previous output only after the entire new file is saved.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(rendered + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
