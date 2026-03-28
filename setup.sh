# curl -LsSf https://astral.sh/uv/install.sh | sh
uv init -p 3.13
uv add deepagents
uv add langchain-openai
uv add python-telegram-bot
uv add pypdf
uv add python-pptx
uv add trafilatura
uv add python-dotenv
rm main.py

