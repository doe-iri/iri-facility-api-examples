import os
import sys
import time
from itertools import islice

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2TokenizerFast

# -------- Config you can tweak for speed/quality --------
NUM_LINES = 1000           # only use first 1000 lines from the big file
VOCAB_SIZE = 2000         # small vocab -> faster training
BLOCK_SIZE = 64          # context length
N_LAYER = 2
N_HEAD = 4
N_EMBD = 128
BATCH_SIZE = 16
MAX_STEPS = 1000           # keep small to stay < ~2 minutes
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
LOG_INTERVAL = 20
# --------------------------------------------------------

torch.manual_seed(42)
torch.set_num_threads(min(8, os.cpu_count() or 1))
device = torch.device("cpu")

def read_first_lines(path, n=NUM_LINES):
    lines = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in islice(f, n):
            if line.strip():
                lines.append(line.rstrip('\n'))
    return lines

def train_byte_level_bpe(lines, save_dir, vocab_size=VOCAB_SIZE):
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = os.path.join(save_dir, "subset.txt")
    with open(tmp_path, "w", encoding="utf-8") as w:
        w.write("\n".join(lines))

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[tmp_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
    )
    tokenizer.save_model(save_dir)  # writes vocab.json and merges.txt

    # Load as a GPT2-compatible tokenizer
    hf_tok = GPT2TokenizerFast.from_pretrained(
        save_dir,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    return hf_tok

class LineConcatDataset(Dataset):
    def __init__(self, tokenizer, lines, block_size=BLOCK_SIZE):
        text = "\n".join(lines)
        # Tokenize into a single stream, then split into fixed-length blocks
        ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
        # Make many samples by sliding window (non-overlapping)
        num_blocks = (len(ids) - 1) // block_size
        self.inputs = []
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            x = ids[start:end]
            if len(x) == block_size:
                self.inputs.append(x)
        # labels = inputs for causal LM (the model does the shift internally)
        self.inputs = torch.stack(self.inputs) if len(self.inputs) > 0 else torch.empty((0, block_size), dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        return {
            "input_ids": x,
            "labels": x.clone(),  # GPT2LMHeadModel shifts internally
            "attention_mask": torch.ones_like(x),
        }

def build_tiny_gpt2(tokenizer, block_size=128, n_embd=128, n_layer=2, n_head=4):
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    try:
        # Newer Transformers versions (v5+) use GPT2ForCausalLM
        from transformers.models.gpt2.modeling_gpt2 import GPT2ForCausalLM as ModelCls
    except Exception:
        # Older v4 versions use GPT2LMHeadModel
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as ModelCls
    return ModelCls(config)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tiny_gpt2_cpu_100lines.py /path/to/large_text.txt")
        sys.exit(1)

    if len(sys.argv) > 2:
         workdir = f"{sys.argv[2]}/tiny_gpt2_artifacts"
    else:
         workdir = "./tiny_gpt2_artifacts"

    print(f"Using {workdir} for output ....")
    
    big_path = sys.argv[1]
    lines = read_first_lines(big_path, NUM_LINES)
    if not lines:
        print("No lines read; check file path or encoding.")
        sys.exit(1)
    
    tokenizer = train_byte_level_bpe(lines, os.path.join(workdir, "tokenizer"))
    model = build_tiny_gpt2(tokenizer).to(device)
    model.train()

    ds = LineConcatDataset(tokenizer, lines, BLOCK_SIZE)
    if len(ds) == 0:
        print("Dataset produced zero samples; try increasing NUM_LINES or reducing BLOCK_SIZE.")
        sys.exit(1)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optim = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Vocab size: {tokenizer.vocab_size} | Samples: {len(ds)} | Steps: {MAX_STEPS}")
    print("Starting training on CPU...")

    step = 0
    start_t = time.time()
    running_loss = 0.0
    while step < MAX_STEPS:
        for batch in dl:
            step += 1
            inputs = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**inputs)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running_loss += loss.item()
            if step % LOG_INTERVAL == 0:
                avg = running_loss / LOG_INTERVAL
                elapsed = time.time() - start_t
                print(f"step {step:4d} | loss {avg:.4f} | {elapsed:.1f}s elapsed")
                running_loss = 0.0

            if step >= MAX_STEPS:
                break

    elapsed = time.time() - start_t
    print(f"Done. Total time: {elapsed:.1f}s")

    # Save model and tokenizer
    save_dir = os.path.join(workdir, "tiny_gpt2_model")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved to: {save_dir}")

    # Quick generation demo
    model.eval()
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.9,
            top_k=50,
        )
    print("Prompt:", prompt)
    print("Output:", tokenizer.decode(gen_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
