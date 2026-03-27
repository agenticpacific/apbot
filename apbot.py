# uv init -p 3.13
# uv add deepagents
# uv add python-telegram-bot
# uv add pypdf
# uv add python-pptx
# uv add trafilatura

import logging, os
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    filters,
    MessageHandler,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from deepagents.backends import LocalShellBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()  # Load environment variables from .env file
CHAT_ID = int(os.environ["CHAT_ID"])
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
USER = os.environ["USER"]

telegram_bot = None

# llama.cpp
llm_model_gguf = ChatOpenAI(
    model="unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL",
    api_key="none",
    base_url="http://localhost:8080/v1",
    use_responses_api=False,
)

# mlx
llm_model_mlx = ChatOpenAI(
    model=str(Path.home()) + "/Models/mlx-community/Qwen3.5-9B-MLX-4bit",
    api_key="none",
    base_url="http://localhost:8080/v1",
    use_responses_api=False,
)

# nvidia
# nvidia_model = "openai/gpt-oss-120b"
nvidia_model = "qwen/qwen3.5-122b-a10b"
# nvidia_model = "qwen/qwen3.5-397b-a17b"
llm_model_nvidia = ChatOpenAI(
    model=nvidia_model,
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    use_responses_api=False,
)


# agent tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# send file tool
async def send_file(filename: str) -> str:
    """Send a file or document."""
    if telegram_bot is None:
        raise RuntimeError("Telegram bot is not initialized.")
    await telegram_bot.send_document(chat_id=CHAT_ID, document=filename)
    return f"Sent File: {filename}"


# agent
memory_check_pointer = MemorySaver()
agent = create_deep_agent(
    model=llm_model_nvidia,
    tools=[send_file],
    backend=LocalShellBackend(
        root_dir=".", env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin"}, virtual_mode=True
    ),
    system_prompt=f"""
    "You are a helpful assistant (Agentic Pacific) that operates in a Telegram chat, specific to a user named {USER}."
    "You operate in a uv Python virtual environment, and can write and execute Python code."
    "You have the following libraries available: pypdf, python-pptx, trafilatura (to retrieve and extract web content), and can install additional libraries using `uv add <library_name>`."
    "Always generate files in the current directory and use relative paths. Never use absolute paths."
    "You can use the following tools: send_file(filename) to send a file or document to the user."
    "Always use the send_file tool to send files to the user instead of printing file contents.
    """,
    store=InMemoryStore(),
    checkpointer=memory_check_pointer,
    # memory=["/AGENTS.md"],
)

# logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


# direct telegram command handler
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a Telegram Bot for Agentic Pacific."
    )


async def process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if (
        update.effective_chat.id == CHAT_ID
    ):  # Only respond to messages from this chat ID (Sachin)
        # Run the agent
        print(f"Req Recieved: {update.message.text}")
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": update.message.text}]},
            config={"configurable": {"thread_id": "user_session_1"}},
        )
        # print(result)
        print(result["messages"][-1].content)
        reply = result["messages"][-1].content

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.message.message_id,
            text=reply,
        )


if __name__ == "__main__":
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    telegram_bot = application.bot

    # handlers
    process_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), process)
    help_handler = CommandHandler("help", help)

    # register handlers and loop
    application.add_handler(help_handler)
    application.add_handler(process_handler)
    application.run_polling()
