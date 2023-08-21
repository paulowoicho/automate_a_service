## Automated Writing Service

This is a simple automated writing service prototype that uses Large Language Models (LLMs) to evalaute the quality of a student's writing based on their topic of interest and a Common Core Standard. It uses llama-2-70B-chat, but can be easily modified to use any LLM available on the HuggingFace model hub.

To run:
- Visit HuggingFace (create an account if you don't have one) and create and copy the API key from your profile into the .env file.

- Create a virtual environment and install the requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Run the app
```bash
python main.py
```

Note that the prototype uses an LLM to simulate a student using the system. Transcripts of the chat will be saved in the transcripts folder.