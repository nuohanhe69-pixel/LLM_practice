from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_experiment.constants import EXPECTED_DEMO_COUNT
from llm_experiment.dataset import DemoSample, TestSample

PROMPT_FILES = {
    "zero_shot": "zero_shot.txt",
    "few_shot": "few_shot.txt",
}


class PromptTemplateError(ValueError):
    """Raised when a prompt type or external template is invalid."""


def render_prompt(
    prompt_type: str,
    sample: TestSample,
    demos: Sequence[DemoSample],
    *,
    prompt_dir: str | Path,
) -> str:
    if prompt_type not in PROMPT_FILES:
        raise PromptTemplateError(f"Unsupported prompt type: {prompt_type!r}")

    directory = Path(prompt_dir)
    label_definitions = _read_template(directory / "label_definitions.txt")
    template = _read_template(directory / PROMPT_FILES[prompt_type])
    demonstrations = ""
    if prompt_type == "few_shot":
        if len(demos) != EXPECTED_DEMO_COUNT:
            raise PromptTemplateError(
                f"Few-shot prompting requires exactly {EXPECTED_DEMO_COUNT} demos"
            )
        demo_template = _read_template(directory / "demo_example.txt")
        demonstrations = "\n\n".join(
            _format_template(
                demo_template,
                number=index,
                error_text=demo.error_text,
                label=demo.label,
            )
            for index, demo in enumerate(demos, start=1)
        )

    return _format_template(
        template,
        label_definitions=label_definitions,
        demonstrations=demonstrations,
        error_text=sample.error_text,
    )


def _read_template(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PromptTemplateError(f"Cannot read prompt template: {path}") from exc
    if not content:
        raise PromptTemplateError(f"Prompt template is empty: {path}")
    return content


def _format_template(template: str, **values: object) -> str:
    try:
        return template.format(**values)
    except (IndexError, KeyError, ValueError) as exc:
        raise PromptTemplateError(f"Invalid prompt template placeholder: {exc}") from exc
