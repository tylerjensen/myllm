# myllm
Experimentation with fine tuning LLMs for use on ollama.

## Setup Python venv
At the parent directory of your code, create a python virtual environment with the included requirements.txt.

```bash
python -m venv myllm-env
source myllm-env/bin/activate
```

Then install the merged dependencies.

```bash
pip install -r ./myllm/requirements.txt
```

Note: You may have to install missing dependencies such as cairo. It depends on your local. I'm running this in Ubuntu and was 
missing cairo and gobject-instrospection-1.0. Just get the requirements installed completely. Some of these you will need installed 
outside of your venv.

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

You many need to install torch from latest stable like this in order to get requirements installed:

```bash
pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## llama.cpp

Now you need to clone https://github.com/ggml-org/llama.cpp into the parent folder of this code. Then in the root folder of that repo, 
create a directory called build and in that directory run cmake. You may have to install cmake.

```bash
cmake ..
cmake --build .
```

That might take a few minutes, so now's a good time for a break.

## analyze.py: ollama, your analysis model, and analyzing your code

In my initial tests, I'm using `llama3.1:8b` and ollama version `0.5.4`. For production run, I might switch to `llama3.3` but that is slower on my A6000 and you might not have enough GPU VRAM to handle that big model. You need to put ollama into serve mode and then run the model.

```bash
ollama serve
ollama run llama3.1:8b
```

Now modify your `src/analyze.py` code to use the specific model you are using. You may want to change the file types you will analyze, 
and modify the path to the code you want to analyze. 

```py
OLLAMA_MODEL = "llama3.1:8b"  # e.g., "mistral", "codellama", etc.
#...
        for file_path in list(self.code_dir.rglob('*.cs')) + list(self.code_dir.rglob('*.ts')) + list(self.code_dir.rglob('*.js')) + list(self.code_dir.rglob('*.html')) + list(self.code_dir.rglob('*.md')):
#...
    csharp_path = '../../FuzzyStrings/src/DuoVia.FuzzyStrings'
```

Now run your analyze.py code.

```bash
python analyze.py
```

The output of analyze.py is the `out/finetune_dataset.jsonl` a JSON Lines formatted file with lines like this:

```json
{
    "prompt": "What are the tests in this code snippet? (in ../../file.cs)", 
    "response": "This is a comprehensive ..."
}
```

## format.py

Now run format.py to convert the pretty JSON into a form that the finetune.py can consume. Yes, I could eliminate this step, 
but I wanted an intermediate step that would allow me to more easily examine the results of the analysis.

```bash
python format.py
```

This will result in the `out/formatted_finetune_dataset.jsonl` file being created.

## finetune.py

You will need a Hugging Face account and access token. Put the toke in a .env file in your src directory and be sure that you have installed python-dotenv in your Python venv.

```ini
HF_TOKEN=hf_******************************
```

Before running finetune.py, be sure nvcc is installed.

```bash
sudo apt install nvidia-cuda-toolkit

nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```

You will also want to install the following:

```bash
pip install torch==2.5.1
pip install transformers peft accelerate
pip install bitsandbytes==0.41.2
```

You may need to run `python -m bitsandbytes` and make sure bitsandbytes can find everything and resolve if not.

In my case, I had to:

```bash
sudo find / -name libcudart.so* 2>/dev/null
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
```

In my case, I had stupidly installed Linux NVIDA drivers in my WSL2 instance. Don't do that.

WSL2 uses the Windows driver. Just download the latest NVIDIA driver for Windows and install it.

If you're on plain old Linux, be sure your NVIDIA drivers for your distro are installed.

I had to reinstall Windows drivers and then purge Linux drivers.

```bash
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get remove --purge cuda-toolkit-12-8
sudo apt-get autoremove

sudo rm -rf /usr/local/cuda*
```

Then in a Windows terminal, kill WSL: `wsl --shutdown`

Then ONLY install CUDA toolkit and NOT drivers.

```bash
sudo apt update
sudo apt install -y cuda-toolkit-12-8

nvidia-smi

Sat Mar 29 14:16:32 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 572.83         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A6000               On  |   00000000:21:00.0  On |                  Off |
| 30%   43C    P8             27W /  300W |    1418MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A              26      G   /Xwayland                             N/A      |
|    0   N/A  N/A              35      G   /Xwayland                             N/A      |
+-----------------------------------------------------------------------------------------+
```

Once you see that view, the `python -m bitsandbytes` command will run successfully.

Not let's try running `python finetune.py`

Had to upgraded existing bitsandbytes:

```bash
pip install bitsandbytes --upgrade
```

To `bitsandbytes-0.45.4`

Then run

```bash
python verify.py 
```

To verify that CUDA is available and your GPU is connected.

Once `verify.py` runs successfully, follow the instructions printed to the console and 
then run the prompt: `What can you tell me about DuoVia.FuzzyStrings?`

And you will get an answer something like this:

```
DuoVia.FuzzyStrings is a library that provides various methods to perform fuzzy matching operations on strings such as Levenshtein Distance, Longest Common Subsequence and more.
The library primarily targets .NET applications and provides the functionality through two interfaces: `IDuoviaStringMatching` and `IDuoviaFuzzyStrings`. The first interface 
defines the core methods required for string matching operations while the second adds additional fuzzy-related methods such as Longest Common Subsequence.
The library also includes an implementation of a Levenshtein Distance algorithm called `DuoViaLevenshteinDistance` which supports both one and two-dimensional arrays. You can find 
```

## Open WebUI

Here's a quick step-by-step tutorial on how to **install and run Open WebUI**, a user interface for interacting with language models like `my_codellama_7b` that we just created:

**Requirements**

- Docker & Docker Compose installed
- Git (optional but useful)
- Ollama (if you want to run models locally like `llama3`, `mistral`, etc.)

**Steps to Install and Run Open WebUI**

1. **Clone the Repo (Optional)**
```bash
git clone https://github.com/open-webui/open-webui.git
cd open-webui
```

2. **Run with Docker (Recommended)**
You can run Open WebUI with Docker using this command:

```bash
docker run -d \
  --name open-webui \
  -p 3000:3000 \
  -e 'OLLAMA_BASE_URL=http://host.docker.internal:11434' \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

This:
- Runs Open WebUI in the background
- Connects to Ollama on your local machine (`http://host.docker.internal:11434`)
- Exposes the web UI on `http://localhost:3000`

3. **Access the UI**
Go to: `http://localhost:3000`


Then run a model like:

```bash
ollama run my_codellama_7b
```

