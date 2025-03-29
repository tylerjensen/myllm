import sys
import os
import json
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  # e.g., "mistral", "codellama", etc.

class CodeAnalyzer:
    def __init__(self, code_dir):
        self.code_dir = Path(code_dir)
        self.dataset = []

    def flatten_ollama_responses(self, lines):
        parsed_responses = []

        for line in lines:
            if not line.strip():
                continue  # Skip empty lines
            data = json.loads(line)
            prompt = data["prompt"]
            response = data["response"]

            # Try to extract file name from original prompt
            if "does" in prompt and "do" in prompt:
                parts = prompt.split("does")
                after_does = parts[1].strip() if len(parts) > 1 else ""
                file_name = after_does.split("do")[0].strip()
            else:
                file_name = ""

            if isinstance(response, str):
                # Single response — keep it as-is
                parsed_responses.append({
                    "prompt": prompt,
                    "response": response.strip()
                })
            elif isinstance(response, list):
                # Nested Q&A — update each with filename in prompt
                for qa in response:
                    sub_prompt = qa["prompt"].strip()
                    if file_name:
                        # Insert file name into the prompt context
                        if "code" in sub_prompt:
                            sub_prompt = sub_prompt.replace("code", f"code in {file_name}")
                        else:
                            sub_prompt = f"{sub_prompt} (from {file_name})"
                    parsed_responses.append({
                        "prompt": sub_prompt,
                        "response": qa["response"].strip()
                    })

        return parsed_responses

    def make_questions_list_prompt(self, file_name: str, content: str):
        prompt = (
            f"Return a JSON object with questions. For the content in a file named {file_name}, generate a list of thoughtful and relevant questions "
            f"that someone might ask to understand or analyze the code. Focus on questions that help uncover the intent, structure, "
            f"design decisions, potential issues, or key patterns in the code."
            f"Here is the file content:\n\n{content}"
        )
        json = {
                 "model": OLLAMA_MODEL,
                 "messages": [
                    { "role": "system", "content": "You are a helpful and concise senior software engineer proficient in Angular, React, HTML, Javascript, Typescript, and C#. \n\n" },
                    { "role": "user", "content": prompt }
                 ],
                 "format": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "questions"
                    ]
                 },
                 "stream": False
               }
        return json

    def make_general_analysis_prompt(self, file_name: str, question: str, content: str):
        prompt = (
            f"For this code with file named {file_name}, generate a detailed answer with "
            f"examples from the code to the following question: {question} "
            f"Here is the file content:\n\n{content}"
        )
        json = {
                 "model": OLLAMA_MODEL,
                 "messages": [
                    { "role": "system", "content": "You are a helpful and concise senior software engineer proficient in Angular, React, HTML, Javascript, Typescript, and C#. \n\n" },
                    { "role": "user", "content": prompt }
                 ],
                 "stream": False
               }
        return json

    def send_to_ollama(self, prompt: str):
        try:
            response = requests.post("http://localhost:11434/api/chat", json=prompt)
            response.raise_for_status()
            #print(f"Status: {response.status_code}")
            print(f"{response.text}")
            content = response.json().get("message", {}).get("content", "")
            #print(f"Content length: {len(content)}")
            return content;
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return f"ERROR: {str(e)}"

    def process_directory(self):
        # Use LLM to analyze the code file
        for file_path in list(self.code_dir.rglob('*.cs')) + list(self.code_dir.rglob('*.ts')) + list(self.code_dir.rglob('*.js')) + list(self.code_dir.rglob('*.html')) + list(self.code_dir.rglob('*.md')):
            filename = str(file_path)
            print(f"filename: {filename}")
            content = file_path.read_text(encoding='utf-8')
            # If response is JSON list, parse and extend
            try:
                prompt = self.make_questions_list_prompt(filename, content)
                model_response = self.send_to_ollama(prompt)
                parsed = json.loads(model_response)
                questions = parsed.get("questions", [])

                print("Extracted Questions:")
                for q in questions:
                    print(f"- {q}")

                for question in questions:
                    prompt = self.make_general_analysis_prompt(filename, question, content)
                    model_response = self.send_to_ollama(prompt)
                    self.dataset.append({
                        "prompt": f"{question} (in {filename})",
                        "response": model_response
                    })

                prompt = self.make_general_analysis_prompt(filename, "What does this code do?", content)
                model_response = self.send_to_ollama(prompt)
                self.dataset.append({
                    "prompt": f"What does the code in {filename} do?",
                    "response": model_response
                })

                prompt = self.make_general_analysis_prompt(filename, "What are the classes defined in this code and what do they do?", content)
                model_response = self.send_to_ollama(prompt)
                self.dataset.append({
                    "prompt": f"What are the classes in {filename} and what do they do?",
                    "response": model_response
                })

                prompt = self.make_general_analysis_prompt(filename, "What are the methods defined in this code and what purpose do they serve?", content)
                model_response = self.send_to_ollama(prompt)
                self.dataset.append({
                    "prompt": f"What are the methods in {filename} and what purpose do they serve?",
                    "response": model_response
                })

                prompt = self.make_general_analysis_prompt(filename, "What are the significant dependencies used in this code and how are they used?", content)
                model_response = self.send_to_ollama(prompt)
                self.dataset.append({
                    "prompt": f"What are the significant dependencies used in {filename} and how are they used?",
                    "response": model_response
                })

                prompt = self.make_general_analysis_prompt(filename, "What is the general quality of the code and how could it be improved?", content)
                model_response = self.send_to_ollama(prompt)
                self.dataset.append({
                    "prompt": f"What is the general quality of {filename} and how could it be improved?",
                    "response": model_response
                })
            except Exception:
                self.dataset.append({
                    "prompt": f"ERR: What does {filename} do?",
                    "response": model_response
                })        

    def save_dataset(self, output_file):
        output_path = Path(output_file)
        with output_path.open('w', encoding='utf-8') as f:
            for item in self.dataset:
                f.write(json.dumps(item) + '\n')

# main
if __name__ == '__main__':
    csharp_path = '../../FuzzyStrings/src/DuoVia.FuzzyStrings'
    analyzer = CodeAnalyzer(csharp_path)
    analyzer.process_directory()
    output_file = Path(__file__).parent / '../out/finetune_dataset.jsonl'
    analyzer.save_dataset(output_file)
