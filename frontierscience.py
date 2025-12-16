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
def olympiad_scorer() -> Scorer:
    """
    Scorer for olympiad-style questions.

    Extracts the final answer from the model output and compares it to the
    expected answer using fuzzy matching to handle LaTeX formatting variations.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Extract the model's final answer
        model_output = state.output.completion

        # Look for "FINAL ANSWER" in the output
        final_answer_pattern = r"FINAL ANSWER[:\s]*(.+?)(?:\n|$)"
        match = re.search(final_answer_pattern, model_output, re.IGNORECASE | re.DOTALL)

        if not match:
            return Score(
                value="C",  # Incorrect
                answer=model_output,
                explanation="No 'FINAL ANSWER' found in output",
            )

        extracted_answer = match.group(1).strip()
        expected_answer = target.text.strip()

        # Normalize answers for comparison
        def normalize_answer(text: str) -> str:
            """Normalize mathematical expressions for comparison."""
            # Remove common LaTeX delimiters
            text = re.sub(r'\\[\[\(]|[\]\)]\\', '', text)
            # Remove backticks
            text = text.replace('`', '')
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove \( and \) delimiters
            text = text.replace('\\(', '').replace('\\)', '')
            return text.strip().lower()

        normalized_extracted = normalize_answer(extracted_answer)
        normalized_expected = normalize_answer(expected_answer)

        # Check for exact match or containment
        is_correct = (
            normalized_extracted == normalized_expected or
            normalized_expected in normalized_extracted or
            normalized_extracted in normalized_expected
        )

        return Score(
            value="C" if is_correct else "I",
            answer=extracted_answer,
            explanation=f"Expected: {expected_answer}\nGot: {extracted_answer}",
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
        grading_prompt = f"""You are a scientific expert tasked with grading a student's answer to a research-level physics question.

ORIGINAL QUESTION:
{state.input_text}

GRADING RUBRIC:
{rubric}

STUDENT'S ANSWER:
{model_output}

Your task is to:
1. Carefully evaluate the student's answer against each item in the rubric
2. Assign points for each rubric item based on how well the answer addresses it
3. Provide a total score and brief justification

Format your response as:
EVALUATION: [Your detailed evaluation of each rubric item]
TOTAL_SCORE: [numerical score out of total possible points]
JUSTIFICATION: [Brief summary of strengths and weaknesses]
"""

        # Get grader's evaluation
        grader_response = await grader.generate(grading_prompt)
        grader_output = grader_response.completion

        # Extract score from grader output
        score_match = re.search(r"TOTAL_SCORE[:\s]*([\d.]+)", grader_output)

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

    # Choose scorer based on category
    if category == "olympiad":
        scorer_fn = olympiad_scorer()
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
