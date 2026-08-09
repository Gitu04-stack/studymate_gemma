from flask import Flask, request, jsonify, send_from_directory
from google import genai
import os

app = Flask(__name__, static_folder="static")

MODEL_NAME = "gemma-4-26b-a4b-it"

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=api_key)


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/ask", methods=["POST"])
def ask_gemma():
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Please enter a question."}), 400

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message
        )

        return jsonify({
            "answer": response.text
        })

    except Exception as e:
        print("Gemma error:", e)
        return jsonify({
            "error": "Gemma could not generate a response."
        }), 500


if __name__ == "__main__":
    app.run()
