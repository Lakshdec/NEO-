#!/usr/bin/env python3
"""
Python AI Chatbot Web Server using Groq API
Serves both API endpoints and a web interface
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder=".", static_folder=".")
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize Groq client - get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please create a .env file with your API key.")
client = Groq(api_key=api_key)

# System prompt for the assistant
SYSTEM_PROMPT = """You are NEO, a helpful, friendly, and knowledgeable AI assistant. 
You provide clear, concise, and accurate responses to user queries. 
You maintain context from previous messages to have meaningful multi-turn conversations.
You are respectful, empathetic, and ready to help with a wide range of topics.
Keep responses conversational and engaging."""

# Store conversation histories per session
conversations = {}


@app.route("/")
def index():
    """Serve the NEO chatbot page"""
    return render_template("chatbot_neo.html")

@app.route("/old")
def old_index():
    """Serve the original chatbot page"""
    return render_template("chatbot.html")


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    """API endpoint for chat messages"""
    if request.method == "OPTIONS":
        return jsonify({"success": True})
    
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        # Initialize or get conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        conversation_history = conversations[session_id]
        
        # Add user message
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Create messages list with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation_history)
        
        # Get response from Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return jsonify({
            "message": assistant_message,
            "session_id": session_id
        })
    
    except RateLimitError:
        return jsonify({"error": "Rate limit exceeded. Please wait a moment."}), 429
    except APIConnectionError as e:
        return jsonify({"error": f"Connection error: {e}"}), 503
    except APIError as e:
        return jsonify({"error": f"API error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat_neo():
    """API endpoint for NEO chat (new frontend)"""
    if request.method == "OPTIONS":
        return jsonify({"success": True})
    
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        conversation_history = data.get("conversation_history", [])
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        # Create messages list with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history
        for msg in conversation_history:
            messages.append({
                "role": msg.get('role', 'user'),
                "content": msg.get('content', '')
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response from Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        
        assistant_message = response.choices[0].message.content
        
        return jsonify({
            "response": assistant_message,
            "success": True
        })
    
    except RateLimitError:
        return jsonify({"error": "Rate limit exceeded. Please wait a moment.", "success": False}), 429
    except APIConnectionError as e:
        return jsonify({"error": f"Connection error: {e}", "success": False}), 503
    except APIError as e:
        return jsonify({"error": f"API error: {e}", "success": False}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}", "success": False}), 500


@app.route("/api/clear", methods=["POST", "OPTIONS"])
def clear_conversation():
    """Clear conversation history"""
    if request.method == "OPTIONS":
        return jsonify({"success": True})
    
    try:
        data = request.json
        session_id = data.get("session_id", "default")
        
        if session_id in conversations:
            conversations[session_id] = []
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    """Health check endpoint"""
    if request.method == "OPTIONS":
        return jsonify({"success": True})
    
    return jsonify({"status": "ok", "service": "NEO Chatbot"})


if __name__ == "__main__":
    print("=" * 60)
    print("NEO - AI Chatbot Server")
    print("=" * 60)
    print("🚀 Server running on http://localhost:5000")
    print("📝 Open chatbot_neo.html in your browser")
    print("=" * 60)
    app.run(debug=True, port=5000)
