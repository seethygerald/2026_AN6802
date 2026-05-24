from flask import Flask, jsonify, render_template, request
import joblib
import os
from groq import Groq
from rag_service import get_equity_rag_service

# api_key = os.getenv("GROQ_API_KEY")

model = joblib.load("foodexp.pkl")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# client = Groq(api_key=api_key)
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return(render_template("index.html"))

@app.route('/main', methods=['GET', 'POST'])
def main():
    return(render_template('main.html'))

@app.route('/ethics', methods=['GET', 'POST'])
def ethics():
    return(render_template('ethics.html'))

@app.route('/correct', methods=['GET', 'POST'])
def correct():
    return(render_template('correct.html'))

@app.route('/wrong', methods=['GET', 'POST'])
def wrong():
    return(render_template('wrong.html'))

@app.route('/econ', methods=['GET', 'POST'])
def econ():
    return(render_template('econ.html'))

@app.route('/foodExp', methods=['GET', 'POST'])
def foodExp():
    q = float(request.form.get("q"))
    r = model.predict([[q]])
    return(render_template("foodExp.html",r=r[0][0]))

@app.route("/chatbot",methods=["get","post"])
def chatbot():
    return(render_template("chatbot.html"))

@app.route('/roe', methods=['GET', 'POST'])
def roe():
    r = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [
            {"role": "system", "content": "Please explain RoE in 20 words"}
        ]
    )
    return(render_template("roe.html",r=r.choices[0].message.content))

@app.route('/generalQuestion', methods=['GET', 'POST'])
def generalQuestion():
    return(render_template("generalQuestion.html"))

@app.route('/groqReply', methods=['GET', 'POST'])
def groqReply():
    q = request.form.get("q")
    r = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [
            { "role": "system", "content": q}
        ]
    )
    return(render_template("groqReply.html",r=r.choices[0].message.content))

@app.route("/equity",methods=["get","post"])
def equity():
    return(render_template("equity.html"))

@app.route("/equity/query", methods=["POST"])
def equity_query():
    payload = request.get_json(silent=True) or {}
    q = (payload.get("q") or request.form.get("q") or "").strip()

    if not q:
        return jsonify({"error": "Question is required."}), 400

    try:
        service = get_equity_rag_service()
        return jsonify(service.ask(q))
    except Exception as e:
        return jsonify({"error": f"Unable to answer right now: {e}"}), 500

@app.route("/apple",methods=["get","post"])
def apple():
    return(render_template("apple.html"))

if __name__ == "__main__":
    app.run()
