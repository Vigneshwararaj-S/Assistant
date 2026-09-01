import threading
from transformers import TextIteratorStreamer
from model import Model

class Chat:
    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.7):
        self.model = Model(model_name)
        self.tokenizer = self.model.tokenizer
        self.history = []
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _generate_reply(self, **kwargs):
        streamer = kwargs["streamer"]
        try:
            self.model.model.generate(**kwargs)
        except RuntimeError as e:
            print(f"\n[Generation failed: {e}]")
            streamer.end()



    def chat(self, prompt: str):
        if prompt.strip().lower() == "/exit":
            print("Exiting chat...")
            return

        if prompt.strip().lower() == "/reset":
            self.history = []
            print("Chat history reset.")
            return

        self.history.append({"role": "user", "content": prompt})

        template = self.tokenizer.apply_chat_template(self.history, 
                                                      tokenize=False, 
                                                      add_generation_prompt=True)


        input_ids = self.tokenizer(template, return_tensors="pt").input_ids.to(self.model.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(input_ids=input_ids, 
                                 streamer=streamer, 
                                 max_new_tokens=self.max_new_tokens, 
                                 temperature=self.temperature,
                                 do_sample=True,)

        thread = threading.Thread(target=self._generate_reply, kwargs=generation_kwargs)
        thread.start()

        print("\nAssistant: ", end="")

        reply = ""
        for new_token in streamer:
            print(new_token, end="", flush=True)
            reply += new_token
        self.history.append({"role": "assistant", "content": reply})



