# from flask import Flask, render_template, request, jsonify
# from model_loader import pipe, tokenizer  # ✅ Shared model
# import budgetsummary2
# import test2
# from flask_cors import CORS



# app = Flask(__name__)
# CORS(app)

# @app.route('/')
# def home():
#     return render_template('home_page.html')

# @app.route('/budget-summary')
# def budget_summary_page():
#     return render_template('budget_summary.html')

# @app.route('/qa')
# def qna_page():
#     return render_template('rag_chatbot.html')

# @app.route('/generate-budget', methods=['POST'])
# def generate_budget():
#     data = request.get_json()
#     if not data:
#         return jsonify({"error": "No data received"}), 400
#     try:
#         result = budgetsummary2.generate_budget_summary(data, pipe, tokenizer)  # ✅ Pass model
#         return jsonify({"summary": result})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/generate-qa', methods=['POST'])
# def generate_qa():
#     data = request.get_json()
#     if not data:
#         return jsonify({"error": "No data received"}), 400

#     persona = data.get("persona")
#     query = data.get("query")
#     if not persona or not query:
#         return jsonify({"error": "JSON must include 'persona' and 'query'"}), 400

#     from test2 import session_user_info
#     for key in ["income", "age", "risk_tolerance", "financial_goals"]:
#         if key in data:
#             session_user_info[key] = data[key]

#     try:
#         result = test2.chat({"persona": persona, "query": query}, pipe, tokenizer)  # ✅ Pass model
#         return jsonify({"response": result})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=False)  # 🚨 Turn off debug to avoid double-loading


from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import test2
from budgetsummary2 import generate_budget_summary
import nluanalysis
import spendinginsights 

app = Flask(__name__)
CORS(app)  # ✅ Allow frontend calls

@app.route('/')
def home():
    return render_template('home_page.html')

@app.route('/qa')
def qna_page():
    return render_template('rag_chatbot.html')

@app.route('/nlu')
def nlu_page():
    return render_template('nlu_analysis.html')

@app.route('/budget-summary')
def budget_summary_page():
    return render_template('budget_summary.html') 

@app.route('/spending-insights')
def spending_page():
    return render_template('spending_insights.html')

@app.route('/generate-qa', methods=['POST'])
def generate_qa():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    result = test2.chat(data)
    return jsonify(result)

@app.route('/generate-nlu', methods=['POST'])
def generate_nlu():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    try:
        result = nluanalysis.nlu_analysis(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/generate-budget', methods=['POST'])
def generate_budget():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    try:
        result = generate_budget_summary(data)
        return jsonify({"summary": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/generate-spending', methods=['POST'])
def generate_spending():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    try:
        result = spendinginsights.spending_insights(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000, debug=False)
