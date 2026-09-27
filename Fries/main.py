import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOLS, TOOL_DEFINITIONS

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
def run_tool(tool_name, arguments):
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    try:
        result = TOOLS[tool_name](**arguments)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Tool error: {str(e)}"
        }
def is_translation_request(message):
    text = message.lower()
    translation_words = [
        "translate",
        "translation",
        "translate this",
        "translate that",
        "in english",
        "in hindi",
        "in french",
        "in spanish",
        "in german",
        "in japanese",
        "in chinese",
        "in korean",
        "in russian",
        "in arabic",
        "in italian",
        "in portuguese",
        "in bengali",
        "in tamil",
        "in telugu",
        "in marathi",
        "in gujarati",
        "in punjabi",
        "in urdu"
    ]
    return any(word in text for word in translation_words)
def ask_ai(user_message):
    conversation = [
        {
            "role": "user",
            "content": user_message
        }
    ]
    force_translation = is_translation_request(user_message)
    first_request = True
    while True:
        if force_translation and first_request:
            tool_choice = {
                "type": "function",
                "name": "translate_text"
            }
        else:
            tool_choice = "auto"
        response = client.responses.create(
            model="openrouter/free",
            input=conversation,
            tools=TOOL_DEFINITIONS,
            tool_choice=tool_choice
        )
        first_request = False
        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]
        if not tool_calls:
            return response.output_text
        conversation.extend(response.output)
        for call in tool_calls:
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = {}
            result = run_tool(
                call.name,
                arguments
            )
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(
                        result,
                        default=str,
                        ensure_ascii=False
                    )
                }
            )

print("          FlexAI")
print("Tools loaded:")
for tool_name in TOOLS:
    print(f"- {tool_name}")
print()
print("Type 'exit' or 'quit' to stop.")
while True:
    user_message = input("\nYou: ")
    if user_message.lower().strip() in ["exit", "quit"]:
        print("FlexAI: Goodbye!")
        break
    if not user_message.strip():
        continue
    try:
        answer = ask_ai(user_message)
        print("FlexAI:", answer)
    except Exception as e:
        print("FlexAI: Error:", e)