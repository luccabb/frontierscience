"""
FrontierScience Eval - Inspect Framework Integration

This eval measures AI's ability to perform scientific research tasks across
two categories: olympiad-level problems and research-level questions.

Paper: https://cdn.openai.com/pdf/2fcd284c-b468-4c21-8ee0-7a783933efcc/frontierscience-paper.pdf
"""

import json
import re
from pathlib import Path
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model, get_model
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import (
    generate,
    Solver,
    TaskState,
    solver,
)


def load_frontierscience_dataset(
    category: Literal["olympiad", "research"] = "olympiad",
    limit: int | None = None,
) -> Dataset:
    """
    Load FrontierScience dataset from JSONL files.

    Args:
        category: Which subset to load - "olympiad", "research"
        limit: Maximum number of samples to load (for testing)

    Returns:
        Dataset containing FrontierScience samples
    """
    data_dir = Path(__file__).parent / "data"
    samples = []

    filepath = data_dir / f"{category}.jsonl"
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(samples) >= limit:
                break

            data = json.loads(line.strip())

            # Create sample with metadata
            sample = Sample(
                input=data["problem"],
                target=data["answer"],
                metadata={
                    "category": category,
                    "subject": data.get("subject", "unknown"),
                    "task_group_id": data.get("task_group_id", ""),
                },
            )
            samples.append(sample)

    return MemoryDataset(samples=samples)


@scorer(metrics=[accuracy(), stderr()])
def olympiad_scorer(grader_model: str | Model | None = None) -> Scorer:
    """
    Scorer for olympiad-style questions using LLM-as-a-judge.

    Uses a grader model to check equivalency of mathematical expressions, numbers, or phrases.

    Note: The paper uses GPT-5 at "high" reasoning effort for grading olympiad questions.

    Args:
        grader_model: Model to use for grading (defaults to openai/gpt-5)
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get the grader model - default to GPT-5 as per paper
        if grader_model is None:
            grader = get_model("openai/gpt-5")
        elif isinstance(grader_model, str):
            grader = get_model(grader_model)
        else:
            grader = grader_model

        model_output = state.output.completion
        expected_answer = target.text

        # Extract everything after "FINAL ANSWER:" or "FINAL ANSWER"
        # Let the LLM judge handle any formatting variations
        final_answer_match = re.search(
            r"FINAL ANSWER\s*:?\s*(.+)",
            model_output,
            re.IGNORECASE | re.DOTALL
        )

        if final_answer_match:
            attempted_answer = final_answer_match.group(1).strip()
        else:
            # If no FINAL ANSWER marker, use the whole output
            attempted_answer = model_output

        # Create grading prompt matching the paper's approach
        grading_prompt = f"""You are grading an attempted answer to a science olympiad problem. You will be given the
problem, attempted answer, and reference answer. Evaluate the solution against the provided
reference solution, ensuring it is complete and matches the reference solution. Pay close
attention to detail and grade it strictly, but fairly.

The reference answer is either a single number or expression in latex formatting, a chemical
formula, a compound name, or a phrase referring to a specific name, entity, or method.

Mark the attempted answer as correct if it fully matches the reference answer or is otherwise
equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal
place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is not
equivalent to the reference answer.

***
The Problem: {state.input_text}
***
The Reference Answer: {expected_answer}
***
The Attempted Answer: {attempted_answer}
***

First, think step-by-step about whether the attempted answer matches the reference answer.
If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your
response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT"
"""

        # Get grader's evaluation
        grader_response = await grader.generate(grading_prompt)
        grader_output = grader_response.completion

        # Extract verdict from grader output
        is_correct = "VERDICT: CORRECT" in grader_output.upper()

        return Score(
            value="C" if is_correct else "I",
            answer=attempted_answer,
            explanation=f"Grader verdict: {'CORRECT' if is_correct else 'INCORRECT'}\n\nExpected: {expected_answer}\nGot: {attempted_answer}\n\nGrader reasoning:\n{grader_output}",
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def research_scorer(grader_model: str | Model | None = None) -> Scorer:
    """
    Scorer for research-level questions using LLM-as-a-judge.

    Uses a grader model to evaluate detailed solutions against rubrics.

    Note: The FrontierScience paper uses GPT-5 at high reasoning effort as the judge model.

    Args:
        grader_model: Model to use for grading (defaults to openai/gpt-5 with high reasoning)
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get the grader model - default to GPT-5 with high reasoning effort as per paper
        # Note: To match the paper exactly, you may want to configure the model with
        # reasoning_effort="high" if your API/framework supports it
        if grader_model is None:
            grader = get_model("openai/gpt-5")
        elif isinstance(grader_model, str):
            grader = get_model(grader_model)
        else:
            grader = grader_model

        model_output = state.output.completion
        rubric = target.text

        # Create grading prompt
        grading_prompt = f"""You are grading a science exam.

You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric
will total up to 10 points.

Evaluate the attemped answer against the provided rubric. Pay close attention to detail and
grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make
any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should
treat the rubric as the gold standard). Return the absolute total number of points earned (it can
be a decimal based on the rubric).

***
The Problem: {state.input_text}
***
The Rubric: {rubric}
***
The Attempted Answer: {model_output}
***

First, think step-by-step about each rubric item. Explain your reasoning for each rubric item.
Then, tally the points up and write VERDICT: <total_points> in the last line of your response,
no other text. For example, VERDICT: 2.5 or VERDICT: 0
"""

        # Get grader's evaluation
        grader_response = await grader.generate(grading_prompt)
        grader_output = grader_response.completion

        # Extract score from grader output (looking for VERDICT: format)
        score_match = re.search(r"VERDICT[:\s]*([\d.]+)", grader_output, re.IGNORECASE)

        # Extract total possible points from rubric
        points_pattern = r"Points:\s*([\d.]+)"
        rubric_points = re.findall(points_pattern, rubric)
        total_possible = sum(float(p) for p in rubric_points) if rubric_points else 10.0

        if score_match:
            awarded_score = float(score_match.group(1))
            # Normalize to 0-1 range for accuracy metric
            normalized_score = min(awarded_score / total_possible, 1.0) if total_possible > 0 else 0.0
        else:
            # If we can't extract a score, default to 0
            normalized_score = 0.0
            awarded_score = 0.0

        return Score(
            value=normalized_score,
            answer=model_output,
            explanation=f"Grader evaluation:\n{grader_output}",
            metadata={
                "awarded_score": awarded_score,
                "total_possible": total_possible,
                "grader_output": grader_output,
            },
        )

    return score


@task
def frontierscience(
    category: Literal["olympiad", "research"] = "olympiad",
    limit: int | None = None,
    grader_model: str | None = None,
) -> Task:
    """
    FrontierScience evaluation task.

    Args:
        category: Which subset to evaluate - "olympiad", "research"
        limit: Maximum number of samples to evaluate (for testing)
        grader_model: Model to use for grading research questions (defaults to openai/gpt-5)
                     Note: Paper uses GPT-5 at "high" reasoning effort

    Returns:
        Inspect Task configured for FrontierScience evaluation
    """
    dataset = load_frontierscience_dataset(category=category, limit=limit)

    # Note: Both olympiad and research use LLM-based grading as per the paper
    if category == "olympiad":
        scorer_fn = olympiad_scorer(grader_model=grader_model)
    elif category == "research":
        scorer_fn = research_scorer(grader_model=grader_model)
    else:
        raise Exception("category has to be one of olympiad, or research")

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=scorer_fn,
        name=f"frontierscience_{category}",
    )
