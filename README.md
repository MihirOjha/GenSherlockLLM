# 🕵️‍♂️ Sherlock HoLLMs — A Fine-Tuned GPT-2 Model on the Sherlock Holmes Canon

I remember the first detective I read about was Sherlock Holmes. I love his analytical mindset (something I would want to incorporate minus being a "high functioning sociopath") and the simple process of elimination he employes to uncover the truth. So I thought, hey, how would Sherlock adivse me on getting me into cooking. But I quickly learnt that ideas are one thing but implementation is another beasy I was not prepared for. Hence, my unpolished, unrefined version of at least getting the model to generate sherlockian style prose. Hope you find something of value from this. If you stumbled upon this project and liked the idea and want to collab, please reach out. I would love feedback and learn so much more.

Sherlock HoLLMs is a lightweight language model fine-tuned on **all Sherlock Holmes novels and short stories by Sir Arthur Conan Doyle**.

The goal is to generate _Holmes-style prose_ — short deductions, philosophical reflections, and Victorian-era storytelling with a Sherlockian tone.

This project demonstrates:

- 📚 Dataset preparation from public-domain literature
- 🧠 Fine-tuning GPT-2 using LoRA adapters (parameter-efficient training)
- ⚙️ A reproducible training pipeline
- 📝 An inference script for generating new Holmes-like text
- 🚀 A complete end-to-end ML project suitable for learning or portfolio work

---

## 🔍 Desired Example Outputs

**Prompt:** _"Holmes once said to Watson about courage and intellect:"_

**Generated:**

> "Holmes once said to Watson about courage and intellect: 'You perceive, my dear fellow, that courage without clarity is merely noise, and intellect without conviction is a lantern without a flame.'"

---

**Prompt:** _"What advice would Sherlock Holmes give a young detective?"_

**Generated:**

> "Apply your reason before your reactions, and cultivate the silence in which truth is most inclined to speak."

Now I wasn't able to acheive this functioanlity because of the model I used and the small dataset. Maybe I could have fine tuned it further but I am still researching on the different ways. If you have any ideas, please feel free to share. I would love for this project to actually mimic Sherlock

---

## 📦 Project Structure

sherlock-wisdom/
├── data/
│ ├── raw/ # raw downloaded text files
│ ├── cleaned/ # processed, chunked data
│ └── manifests/ # metadata, logs
├── src/
│ ├── data_prep/
│ │ └── preprocess.py
│ ├── training/
│ │ └── train_lora.py
│ ├── eval/
│ │ └── overlap_check.py
│ ├── api/
│ │ └── app.py
│ ├── scheduler/
│ │ └── poster.py
│ └── utils/
│ └── helpers.py
├── experiments/ # where training runs/logs are stored
├── venv/ # your virtual environment
├── requirements.txt # dependencies
├── README.md
└── .gitignore

---

## 🧠 Model Architecture

This project fine-tunes:

- **Base Model:** GPT-2 (small)
- **Training Method:** LoRA (Low-Rank Adaptation)
- **Framework:** Hugging Face Transformers
- **Epochs:** 3
- **Sequence Length:** 512
- **Precision:** FP16 (if GPU supports it)

LoRA allows the model to learn stylistic patterns _without_ updating all GPT-2 parameters — making training fast, lightweight, and accessible on consumer GPUs.

---

## 📚 Dataset

All Sherlock Holmes works included here are in the **public domain**.  
Texts were:

- cleaned
- split into paragraphs
- tokenized
- formatted as plain text for causal LM training

No validation split was used (small corpus + educational project).

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/SherlockWisdom.git
cd SherlockWisdom
python -m venv venv
source venv/bin/activate   # or Scripts\activate on Windows
pip install -r requirements.txt
```

## Training

To fine-tune the model from scratch (optional):

python src/train/train_lora.py

This script will:

- Load GPT-2
- Apply LoRA adapters
- Tokenize the Sherlock dataset
- Train for 3 epochs
- Save the LoRA adapter in experiments/sherlock_lora/

## Inference

Generate Holmes-style text with:

python src/eval/infer.py --prompt "Your prompt here"

Example:

python src/eval/infer.py --prompt "Watson was shocked when Sherlock"

## Output

🔍 Loading fine-tuned model...

🧠 Sherlock Wisdom:

Watson was shocked when Sherlock Holmes arrived at the door. He had come upon us for some other reason, and could not have expected to find any sign of this man who he knew so well as I did!” “I am very sorry that you are now in such a bad condition; but it is possible we may be safe here." Watson looked away from him with an expression which seemed rather like resignation: "You will think better if me myself should ask your opinion about these matters before my departure arrives later today--if only after one hour or two!" Then his eyes were full again on our conversation-table—an extraordinary sight among them all alike except Mr., whom they both admired admirably since their first meeting together twenty years ago between Brother John Hawkins and Drs.-John Wardleton (who would soon become Mrs.—Papa) Hallenbeck's great companion until she died last week.]

## 🚀 Deployment

This is originally what I had in mind before starting the porject but this is so basic that I do not know what more can I do with this.

- A Twitter/X bot posting “Holmes Wisdom” daily
- A Gradio web UI for interactive generation
- A VS Code extension that rewrites text in Sherlock’s voice
- A small API endpoint (FastAPI) for programmatic use

## Future Improvements

- Add a validation split and perplexity evaluation
- Fine-tune on a larger base model (GPT-Neo or GPT-J)
- Use RLHF or reward models to improve quote-style responses
- Add prompt templates for cleaner outputs
- Train on additional Victorian literature for richer consistency

## 📜 License

- All Sherlock Holmes texts used here are in the public domain.
- Model code is MIT-licensed.
- Model outputs may resemble Conan Doyle’s style but are not copied text.

## 🙌 Acknowledgments

- Sir Arthur Conan Doyle — for the timeless detective himself
- Hugging Face — for Transformers, Datasets, and PEFT
- The open-source ML community

If you are with me so far, reach out and lets build something together.
