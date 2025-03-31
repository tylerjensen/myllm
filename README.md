# How to Fine Tune a Private LLM on Your Legacy Code Base

Fine-tuning a Large Language Model (LLM) on your existing code base can significantly boost the model’s relevance and performance for your unique domain. This guide walks you through the complete process of analyzing your legacy code, generating a fine-tuning dataset, and training a model using [Ollama](https://ollama.com/) and tools like `llama.cpp`.

Whether you're working with C#, TypeScript, HTML, Javascript, or Markdown, this experimental tutorial helps you convert your source into meaningful prompts and responses, format the data for training, and fine-tune your own private LLM.

It is far from a completed solution. Think of it as simply a starting point.

---
## myllm
---

## Step 1: Set Up a Python Virtual Environment

First, create a Python virtual environment at the root of your project directory.

```bash
python -m venv myllm-env
source myllm-env/bin/activate
```

Then install the required dependencies:

```bash
pip install -r ./myllm/requirements.txt
```

### Additional Setup for Ubuntu

Some dependencies must be installed on your system (not just the virtual environment):

```bash
sudo apt update
sudo apt install libcairo2-dev
sudo apt install libgirepository1.0-dev gir1.2-glib-2.0
sudo apt install cloud-init
sudo apt install command-not-found
sudo apt install libdbus-1-dev libglib2.0-dev
sudo apt install distro-info
sudo apt install python3-apt
sudo apt install libsystemd-dev pkg-config
sudo apt install ubuntu-pro-client
sudo apt install unattended-upgrades
```

If needed, install PyTorch manually:

```bash
pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

---

## Step 2: Set Up `llama.cpp`

Clone the [`llama.cpp`](https://github.com/ggml-org/llama.cpp) repo into your project’s parent directory. Then build it with CMake:

```bash
cmake ..
cmake --build .
```

> Note: You may need to install CMake if it's not already available on your system.

---

## Step 3: Analyze Your Code with `analyze.py`

This step uses Ollama to run your model and generate prompt/response pairs from your legacy code.

Make sure Ollama is running:

```bash
ollama serve
ollama run llama3.1:8b
```

Edit `src/analyze.py` to specify:
- Your Ollama model (e.g., `llama3.1:8b`)
- The file extensions you want to scan
- The path to your source code

```py
OLLAMA_MODEL = "llama3.1:8b" # e.g., "mistral", "codellama", etc.
#...
        for file_path in list(self.code_dir.rglob('*.cs')) + list(self.code_dir.rglob('*.ts')) + list(self.code_dir.rglob('*.js')) + list(self.code_dir.rglob('*.html')) + list(self.code_dir.rglob('*.md')):
#...
    csharp_path = '../../FuzzyStrings/src/DuoVia.FuzzyStrings'
```

Now run the analyzer:

```bash
python analyze.py
```

This will generate a `out/finetune_dataset.jsonl` file with prompt-response pairs like:

```json
{
    "prompt": "What are the tests in this code snippet? (in ../../file.cs)", 
    "response": "This is a comprehensive ..."
}
```

---

## Step 4: Format the Data with `format.py`

Convert the raw prompt-response output into a more structured format:

```bash
python format.py
```

This produces:  
`out/formatted_finetune_dataset.jsonl`

---

## Step 5: Fine-Tune Your Model with `finetune.py`

To begin fine-tuning, make sure you have:

- A Hugging Face account and access token
- `python-dotenv` installed in your environment

Create a `.env` file in `src/` with:

```ini
HF_TOKEN=hf_******************************
```

Ensure your system has `nvcc` (CUDA compiler) installed:

```bash
sudo apt install nvidia-cuda-toolkit

nvcc --version
```

Install the training dependencies:

```bash
pip install torch==2.5.1
pip install transformers peft accelerate
pip install bitsandbytes==0.41.2
```

If you run into issues with `bitsandbytes`, make sure the required libraries are discoverable:

```bash
sudo find / -name libcudart.so* 2>/dev/null
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
```

### Important Note for WSL2 Users

If you're using WSL2, **do not install** Linux NVIDIA drivers. Instead, use the Windows driver and:

1. Remove conflicting Linux drivers:

```bash
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get remove --purge cuda-toolkit-12-8
sudo apt-get autoremove
sudo rm -rf /usr/local/cuda*
```

2. Restart WSL:

```bash
wsl --shutdown
```

3. Reinstall only the CUDA toolkit:

```bash
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

4. Confirm setup:

```bash
nvidia-smi
```

Once your environment is CUDA-ready, test `bitsandbytes`:

```bash
python -m bitsandbytes
```

Upgrade if necessary:

```bash
pip install bitsandbytes --upgrade
```

Then run:

```bash
python verify.py 
```

This script checks if CUDA and your GPU are accessible.

Once verified, you're ready to fine-tune:

```bash
python finetune.py
```

You can now interact with your model:

> **Prompt:** What can you tell me about DuoVia.FuzzyStrings?

**Sample Response:**

```
DuoVia.FuzzyStrings is a library that provides various methods to perform fuzzy matching operations on strings such as Levenshtein Distance, Longest Common Subsequence and more...
```

---

## Step 6: Use Open WebUI for a Friendly Interface

Open WebUI provides a graphical front-end to interact with your fine-tuned model.

### Prerequisites

- Docker & Docker Compose
- Git (optional)
- Ollama installed

### Install Open WebUI

1. Clone the repository (optional):

```bash
git clone https://github.com/open-webui/open-webui.git
cd open-webui
```

2. Run the container:

```bash
docker run -d \
  --name open-webui \
  -p 3000:3000 \
  -e 'OLLAMA_BASE_URL=http://host.docker.internal:11434' \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

This will:
- Launch the web app at `http://localhost:3000`
- Connect to your Ollama instance

3. Start your model:

```bash
ollama run my_codellama_7b
```

Now you can interact with your model directly from the browser.

---

Feel free to customize each step based on your environment, codebase, and model preferences. Happy fine-tuning!

---

### Specs

A few people have asked about the machine I'm using. Here's a few details.

AMD Ryzen Threadripper PRO 5975WX 32-Cores, 3600 Mhz. 128GB RAM, NVIDIA RTX A6000 with 48GB VRAM. 

You could do it on larger repos or document repos. Preparing the data is key. I recommend you experiment with smaller datasets to begin and refine iteratively.

Larger data sets would produce better results but require more effort to create the fine tuned set, and, of course, you would wabt to stream those into the fine tuning algorithm rather than reading it all into memory. For larger, production ready results, I'd probably want a system with 256GB RAM and an NVIDIA A100 with 80GB VRAM.
