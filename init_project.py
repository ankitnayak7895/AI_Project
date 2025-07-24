import os
from pathlib import Path

project_name = "AI_Project"

list_of_files = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    ".env",
    "LICENSE",
    "app.py",

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/ingest/__init__.py",
    f"src/{project_name}/ingest/document_loader.py",
    f"src/{project_name}/ingest/embed_documents.py",
    f"src/{project_name}/ingest/init_db.py",

    f"src/{project_name}/retriever/__init__.py",
    f"src/{project_name}/retriever/search.py",
    f"src/{project_name}/retriever/utils.py",

    f"src/{project_name}/llm/__init__.py",
    f"src/{project_name}/llm/gpt_qa.py",
    f"src/{project_name}/llm/prompt_templates.py",

    f"src/{project_name}/ui/__init__.py",
    f"src/{project_name}/ui/streamlit_app.py",
    f"src/{project_name}/ui/components.py",

    f"src/{project_name}/audio/__init__.py",
    f"src/{project_name}/audio/speech_to_text.py",
    f"src/{project_name}/audio/text_to_speech.py",

    f"src/{project_name}/logs/chat_history.json",

    "data/.gitkeep",
    "notebooks/exploration.ipynb",
    "init_setup.sh",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
    else:
        print(f"Skipping existing file: {filepath}")