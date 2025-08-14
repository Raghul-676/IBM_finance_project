# nluanalysis.py
import json
from model_loader import pipe, tokenizer  # use the already-loaded Granite model

def nlu_analysis(user_input):
    """
    Expects: user_input = {"text": "User's query here"}
    Returns: JSON with sentiment, keywords, and entities
    """
    system_msg = (
        "You are an NLU assistant. "
        "Analyze the user's input and respond ONLY with a JSON object containing exactly these keys: "
        "'sentiment' (one of 'positive', 'neutral', 'negative'), "
        "'keywords' (list of important keywords), "
        "'entities' (list of identified entities). "
        "Do NOT include any other text outside the JSON."
    )

    user_msg = json.dumps(user_input)

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
        max_new_tokens=400,
        do_sample=False,
        temperature=0.0,
        return_full_text=False
    )[0]["generated_text"]

    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON output from model", "raw_output": out}


# -----------------------
# Test
# -----------------------
if __name__ == "__main__":
    sample_input = {
        "text": "I'm finding it hard to save money each month. Can you help me manage my spending better?"
    }

    result = nlu_analysis(sample_input)
    print(json.dumps(result, indent=4))
