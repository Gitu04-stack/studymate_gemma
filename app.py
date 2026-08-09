import os
from flask import Flask, request, jsonify, send_from_directory
from google import genai

app = Flask(__name__, static_folder="static")

MODEL_NAME = "gemma-4-26b-a4b-it"

client = genai.Client(
    api_key=os.environ.get("GOOGLE_API_KEY")
)


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/ask", methods=["POST"])
def ask_gemma():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Please enter a question."}), 400

        prompt = f"""
You are StudyMate, a friendly AI study assistant.

Help students understand academic topics clearly and simply.

User's question:
{user_message}

Give a useful, accurate and student-friendly answer.
Use examples when helpful.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return jsonify({
            "answer": response.text
        })

    except Exception as e:
        return jsonify({
            "error": "Something went wrong while contacting Gemma."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
