#!/usr/bin/env python3
"""
Python AI Chatbot using Groq API
Supports multi-turn conversations with error handling
"""
import os
import sys
from dotenv import load_dotenv
from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError


def main():
    """Main function to run the chatbot"""
    
    # Initialize the Groq client
    try:
        # Paste your real, fresh Groq API key inside the quotes below
        load_dotenv()
        api_key = os.getenv('CHATBOT_API_KEY')
        
        client = Groq(api_key=api_key)
    except APIError as e:
        print(f"Error: Invalid API key or authentication failed: {e}")
        sys.exit(1)
    
    # System prompt for the assistant
    system_prompt = """You are a helpful, friendly, and knowledgeable AI assistant. 
You provide clear, concise, and accurate responses to user queries. 
You maintain context from previous messages to have meaningful multi-turn conversations.
You are respectful, empathetic, and ready to help with a wide range of topics."""
    
    # Initialize conversation history
    conversation_history = []
    
    print("=" * 60)
    print("NEO")
    print("=" * 60)
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 60)
    print()
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\nAssistant: Goodbye! Have a great day!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Add user message to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Create messages list with system prompt
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            
            # Get response from Groq API
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                )
                
                assistant_message = response.choices[0].message.content
                
                # Add assistant response to conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # Print the response
                print(f"\nAssistant: {assistant_message}\n")
                
            except RateLimitError:
                print("\nError: Rate limit exceeded. Please wait a moment before sending another message.\n")
                conversation_history.pop()
            
            except APIConnectionError as e:
                print(f"\nError: Connection issue occurred: {e}")
                print("Please check your internet connection and try again.\n")
                conversation_history.pop()
            
            except APIError as e:
                print(f"\nError: API error occurred: {e}\n")
                conversation_history.pop()
        
        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye! Have a great day!")
            break
        
        except Exception as e:
            print(f"\nUnexpected error: {e}\n")


if __name__ == "__main__":
    main()
