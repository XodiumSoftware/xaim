import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import spacy


class SpaCyTokenizedDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Dataset that uses spaCy to segment text into sentences,
    then uses the transformers tokenizer to produce model inputs.
    """
    def __init__(self, filepath: str, tokenizer: PreTrainedTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        self.sentences = [str(sent) for sent in self.nlp(text).sents]
        self.examples = []

        for sent in self.sentences:
            tokens = tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=block_size,
                padding="max_length",
            )
            self.examples.append({
                "input_ids": tokens["input_ids"].squeeze(0),
                "labels": tokens["input_ids"].squeeze(0),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def train(
    model: PreTrainedModel,
    dataset: Dataset[dict[str, torch.Tensor]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> None:
    dataloader = DataLoader(dataset, batch_size=1)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
        if step >= 100:
            break


if __name__ == "__main__":
    model_name = "openai-community/gpt2"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = SpaCyTokenizedDataset("data/train.txt", tokenizer, block_size=32)
    train(model, dataset, tokenizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
