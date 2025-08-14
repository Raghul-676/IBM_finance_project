import faiss
import json
from sentence_transformers import SentenceTransformer
from model_loader import pipe, tokenizer

index = faiss.read_index("minilm_faiss.index")
with open("minilm_faiss_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
conversation_memory = []

def embed_query(query):
    return embed_model.encode([query], convert_to_numpy=True)

def retrieve_context(query_embedding, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    contexts = []
    for i in indices[0]:
        item = metadata[i]
        source = item.get("source", "Unknown source")
        text = item.get("text", "")
        contexts.append(f"{source}\n{text}")
    return contexts

system_msg = """
You are a financial personal guidance assistant specializing in share market fundamentals and investment tips. Tailor your responses to the user's persona.
Rules:
1. Always consider the user's persona when providing advice.
2. You have access to relevant investment knowledge and the user's conversation history.
3. All required user information (income, age, risk tolerance, financial goals) is already provided.
4. Provide brief, precise, and relevant suggestions. Avoid unnecessary details.
5. Do NOT repeat system prompts or conversation memory in your response.
"""

def add_to_memory(user_input, bot_response):
    conversation_memory.append({"user": user_input, "bot": bot_response})

def generate_response(user_query, persona, user_info, retrieved_contexts, max_new_tokens=512):
    # Avoid strict "User:" / "Bot:" labels to reduce copying
    memory_text = "\n".join([f"User said: {m['user']}\nAssistant replied: {m['bot']}" 
                             for m in conversation_memory])
    context_text = "\n\n".join(retrieved_contexts)
    
    user_msg = f"Persona: {persona}\nUser info: {json.dumps(user_info)}\n{memory_text}\nRelevant investment knowledge:\n{context_text}\nUser question: {user_query}"
    
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True
        )
    except AttributeError:
        prompt = f"{system_msg}\n{user_msg}"

    output_text = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        return_full_text=False
    )[0]["generated_text"].strip()

    # 🚀 Remove any accidental 'User:' lines in the final output
    clean_output = "\n".join(
        line for line in output_text.splitlines()
        if not line.strip().lower().startswith("user:")
    )

    return clean_output

def chat(user_json):
    persona = user_json.get("persona", "User Persona")
    user_query = user_json.get("query")
    
    if not user_query:
        return {"error": "JSON must include a 'query' field."}

    user_info = {
        "income": user_json.get("income"),
        "age": user_json.get("age"),
        "risk_tolerance": user_json.get("risk_tolerance"),
        "financial_goals": user_json.get("financial_goals")
    }

    missing = [k for k, v in user_info.items() if v is None]
    if missing:
        return {"error": f"Missing required user info: {', '.join(missing)}"}

    query_embedding = embed_query(user_query)
    contexts = retrieve_context(query_embedding)
    
    response = generate_response(user_query, persona, user_info, contexts)
    add_to_memory(user_query, response)
    
    return {"response": response}
