from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


class Model:
    def __init__(self, model_name: str):
        print(f"Loading {model_name} (first run downloads the weights, this happens once)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")



