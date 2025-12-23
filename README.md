# FrontierScience Eval

Inspect AI evaluation for the FrontierScience benchmark, measuring AI's ability to perform scientific research tasks.

- Paper: https://cdn.openai.com/pdf/2fcd284c-b468-4c21-8ee0-7a783933efcc/frontierscience-paper.pdf
- Blog: https://openai.com/index/frontierscience/
- Hugging Face: https://huggingface.co/datasets/openai/frontierscience

## Quick Start

**Minimum steps to run:**
1. Activate virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Set API key: `export OPENAI_API_KEY=your_key`
4. Run: `inspect eval frontierscience.py --model openai/gpt-4o`
5. View results: `inspect view`

### 1. Activate Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

**For OpenAI:**
```bash
export OPENAI_API_KEY=your_api_key_here
```

**For Anthropic:**
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

**For OpenRouter (access 500+ models):**
```bash
export OPENROUTER_API_KEY=your_openrouter_api_key
```

### 4. Run the Eval

**Basic usage (defaults to Olympiad questions only):**
```bash
inspect eval frontierscience.py --model openai/gpt-4o --reasoning-effort=high
```

**Research questions only:**
```bash
inspect eval frontierscience.py --model openai/gpt-4o -T category=research --reasoning-effort=high
```

**With custom grader model:**
```bash
inspect eval frontierscience.py --model openai/gpt-4o -T grader_model=openai/gpt-4o --reasoning-effort=high
```

**Test run (first 5 samples):**
```bash
inspect eval frontierscience.py --model openai/gpt-4o -T category=olympiad -T limit=5 --reasoning-effort=high
```

## Viewing Evaluation Results

After running an evaluation, Inspect saves logs to the `logs/` directory. View them using:

**Launch log viewer (web interface):**
```bash
inspect view
```

This opens a web browser at http://localhost:7575 where you can:
- View detailed results for each sample
- See model responses and scoring decisions
- Filter and sort by score
- Inspect metadata and errors

**Note:** Logs are stored in `.eval` format (binary) by default. You can also export as JSON.

## Dataset Structure

- `data/olympiad.jsonl` - 100 olympiad-level physics problems
- `data/research.jsonl` - 60 research-level physics questions

## Scoring

- **Olympiad**: LLM-as-judge checking equivalency of expressions/numbers/phrases (defaults to GPT-5 as grader)
- **Research**: LLM-as-judge using detailed rubrics (defaults to GPT-5 as grader)
  - Note: The paper uses GPT-5 at "high" reasoning effort for grading **both** olympiad and research questions
  - You can specify the same via inspect by passing `--reasoning-effort=high`, see: https://inspect.aisi.org.uk/reasoning.html#reasoning-effort
  - You can specify a different grader model with `-T grader_model=model_name`

## Results

Some of the results when running this codebase against OpenAI API models, the CMD used to run was:
```bash
inspect eval frontierscience.py --model <model> -T category=<category> --reasoning-effort=high --max-connections 6
```

### Olympiad

  | Model                               | Our Results | Reference Results |
  |-------------------------------------|-------------|-------------------|
  | GPT-5.2 (openai/gpt-5.2-2025-12-11) | 71%         | 77.1%             |

### Research

  | Model                               | Our Results | Reference Results |
  |-------------------------------------|-------------|-------------------|
  | GPT-5 (openai/gpt-5-2025-08-07)     | 20%         | 25.2%             |