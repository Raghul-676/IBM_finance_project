# spendinginsights.py
import json          # For strict JSON handling
import json5         # For forgiving JSON parsing
from model_loader import pipe, tokenizer  # reuse loaded Granite model

# -----------------------
# Spending insights function
# -----------------------
def spending_insights(user_input):
    """
    Expects: user_input = {"income": ..., "expenses": {...}}
    Returns: JSON with financial insights (summary, spending pattern, tips, etc.)
    """

    # Strict JSON prompt instructions
    system_msg = (
        "You are a professional financial analyst. "
        "Analyze the provided monthly income and expenses, and respond ONLY with a valid JSON object "
        "containing exactly these keys:\n"
        "summary, spending_pattern, budget_health_check, benchmarks, goal_reallocation, actionable_insights, recommendation.\n"
        "Keys must be strings with double quotes, values must be valid JSON types. "
        "Do NOT include any markdown, headings, or explanations outside the JSON."
    )

    user_msg = json.dumps(user_input)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    # Use the chat template exactly as before
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate model output using shared pipe from model_loader
    out = pipe(
        prompt,
        max_new_tokens=1000,
        do_sample=False,
        temperature=0.0,
        return_full_text=False
    )[0]["generated_text"]

    # Attempt strict JSON parsing first, then fallback to json5
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        try:
            return json5.loads(out)
        except Exception:
            return {"error": "Invalid JSON output from model", "raw_output": out}


# -----------------------
# Test
# -----------------------
if __name__ == "__main__":
    sample_input = {
        "income": 6000,
        "expenses": {
            "rent": 1800,
            "groceries": 600,
            "transport": 300,
            "dining_out": 450,
            "utilities": 250,
            "subscriptions": 120,
            "shopping": 550,
            "miscellaneous": 180
        }
    }

    print("Generating spending insights...")
    result = spending_insights(sample_input)
    print(json.dumps(result, indent=4))
