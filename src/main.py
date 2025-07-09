from typing import cast
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import spacy

from utils import Utils


class SentenceTokenizedDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Dataset that uses spaCy to segment text into sentences,
    then uses the transformers tokenizer to produce model inputs.
    """
    def __init__(self, url: str, tokenizer: PreTrainedTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
        self.nlp.add_pipe("sentencizer")

        text: str = Utils.getRepo(url)

        self.sentences = [str(sent) for sent in self.nlp(text).sents]
        self.examples: list[dict[str, torch.Tensor]] = []

        for sent in self.sentences:
            tokens = tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=block_size,
                padding="max_length",
            )
            input_ids = cast(torch.Tensor, tokens["input_ids"])
            self.examples.append({
                "input_ids": input_ids.squeeze(0),
                "labels": input_ids.squeeze(0),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


class Trainer:
    """
    Handles training of a causal language model using a tokenized dataset.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset[dict[str, torch.Tensor]],
        device: torch.device,
        batch_size: int = 1,
        lr: float = 5e-5,
        max_steps: int = 100,
    ):
        self.model = model.to(device)  # type: ignore
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.max_steps = max_steps

        self.dataloader = DataLoader(dataset, batch_size=self.batch_size)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)  # type: ignore

    def train(self) -> None:
        self.model.train()
        for step, batch in enumerate(self.dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step() # type: ignore
            self.optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")
            if step >= self.max_steps:
                break

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)  # type: ignore
        self.tokenizer.save_pretrained(output_dir)  # type: ignore

def test_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, device: torch.device):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50) # type: ignore
    print("Prompt:", prompt)
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True)) # type: ignore


if __name__ == "__main__":
    model_name = "openai-community/gpt2"
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_name))  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token  # type: ignore
    model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(model_name))  # type: ignore

    dataset = SentenceTokenizedDataset("https://github.com/XodiumSoftware/xbim", tokenizer, block_size=32) #TODO: set correct path.
    trainer = Trainer(model, tokenizer, dataset, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train()
    trainer.save("output_model")

    # Load the trained model for testing
    trained_model = AutoModelForCausalLM.from_pretrained("output_model").to(trainer.device) # type: ignore
    trained_tokenizer = AutoTokenizer.from_pretrained("output_model") # type: ignore
    test_model(trained_model, trained_tokenizer, prompt="The xbim project", device=trainer.device) # type: ignore