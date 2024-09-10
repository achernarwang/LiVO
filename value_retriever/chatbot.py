import re
import tiktoken
from openai import OpenAI, RateLimitError, APITimeoutError
from time import sleep
 
class ChatBot:
    def __init__(self, system:str, examples:list[dict]=None, model='gpt-3.5-turbo', base_url=None):
        self.messages = []
        if system is not None:
            self.messages.append({"role": "system", "content": system})
        if examples is not None:
            self.messages.extend(examples)
        self.model = model
        self.chat_start_idx = len(self.messages)
        self.token_limits = 4096

        self.client = OpenAI(base_url=base_url)

        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def get_token_num(self, messages):
        token_num = 0
        for message in messages:
            token_num += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                token_num += len(self.encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    token_num += -1  # role is always required and always 1 token
        return token_num
            
    def cutoff_history(self, reserve_token_num):
        token_num = self.get_token_num(self.messages) + 2  # every reply is primed with <im_start>assistant

        while self.token_limits - token_num < reserve_token_num:
            idx = int((self.chat_start_idx + len(self.messages) -1) / 2)
            message = self.messages.pop(idx)
            token_num -= self.get_token_num([message])
    
    def clear_history(self):
        self.messages = self.messages[:self.chat_start_idx]

    def rollback_last_round(self):
        assert self.messages[-1]["role"] == "assistant"
        self.messages.pop()
        self.messages.pop()

    def chat(self, input_message:str, reserve_tokens=400, max_tokens=2000, temperature=0, presence_penalty=0, memorize=True, timeout=5, max_retry=10) -> str:
        self.messages.append({"role":"user", "content": input_message})

        self.cutoff_history(reserve_tokens)

        err_cnt = 0
        while err_cnt < max_retry:
            try:
                response = self.client.chat.completions.create(
                    model = self.model,
                    temperature = temperature,
                    presence_penalty = presence_penalty,
                    messages = self.messages,
                    timeout=timeout,
                    max_tokens=max_tokens
                )
                response = response.to_dict()
                break
            except RateLimitError as e:
                match = re.search(r"Please retry after (\d+) second", e._message)
                number = int(match.group(1))
                # print(f"sleeping for {number} seconds")
                sleep(number)
            except APITimeoutError as e:
                err_cnt += 1
                print(e)
                print('Time out, retrying...')
            except Exception as e:
                err_cnt += 1
                print(e)
                print(f"sleeping for 10 seconds")
                sleep(10)
        
        if err_cnt >= max_retry:
            raise ConnectionError('Failed to get connected with API')

        content = response['choices'][0]['message']['content'].strip()
        finish_reason = response['choices'][0]['finish_reason']
        if memorize:
            self.messages.append({"role": "assistant", "content": content})
        else:
            self.messages.pop()
        return content, finish_reason

    