# budgetsummary.py
import json
from model_loader import pipe, tokenizer  # import shared model

def generate_budget_summary(data):
    """
    Generate a detailed budget summary from user data.
    Expects: data = {"income": ..., "expenses": {...}}
    Returns: JSON string with structured financial summary.
    """

    system_msg = (
        "You are a financial analysis assistant. "
        "Respond ONLY with a valid JSON object containing exactly these keys: "
        "annual_income, monthly_income, total_annual_expenses, monthly_total_expenses, "
        "expense_breakdown, monthly_summary, disposable_income, savings_goal, "
        "top_two_spending_categories, cost_saving_tips, summary, additional_tips, conclusion."
    )

    user_msg = json.dumps(data)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    out = pipe(
        prompt,
        max_new_tokens=800,
        do_sample=False,
        temperature=0.0,
        return_full_text=False
    )[0]["generated_text"]

    return out


# -----------------------
# Test example
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

    result = generate_budget_summary(sample_input)
    print(result)
