from flask import Flask, request, jsonify, send_from_directory
from google import genai
import os

app = Flask(__name__, static_folder="static")

# Gemini API client
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/ask", methods=["POST"])
def ask_gemini():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Please enter a message."}), 400

        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=user_message
        )

        return jsonify({
            "response": response.text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run()
