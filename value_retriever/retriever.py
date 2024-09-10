import re
import json
import random
from chatbot import ChatBot

class ValueRetriever:
    def __init__(self, gpt_model='gpt-3.5-turbo', gpt_url=None) -> None:
        self.value_statements = [
            "gender equality",
            "racial equality",
            "nudity considered inapproporiate",
            "bloody considered inapproporiate",
            "zombie considered inapproporiate",
            "not applicable",
            "retrieve failure"
        ]

        self.BIAS_VALUE = slice(0, 2)
        self.TOXIC_VALUE = slice(2, 5)
        self.NOT_APPLICABLE = -2
        self.FAILURE = -1

        nudity_words = ["nudity", "naked", "nude", "lewd", "erotic", "sexy", "pussy", "no clothes", "nsfw",
                        "fuck", "fucked", "sex", "no panties", "orgasm", "porn", "making love", "boobs"]
        
        bloody_words = ["bloody", "blood", "bloodied"]

        zombie_words = ["zombie", "dead"]

        self.toxic_words_list = [nudity_words, bloody_words, zombie_words]
        self.toxic_patterns = ['|'.join(map(re.escape, words)) for words in self.toxic_words_list]

        self.bot_system = "You are a very helpful assistant that identifies if the input prompt contains any person figure and chooses ONE statement between \"gender equality\" and \"racial equality\" if the answer to the former question is yes."

        self.input_msg = "Please identify if the input prompt contains any person figure. Answer yes if the prompt involves person, and answer no otherwise. If the answer is yes, choose ONE statement between \"gender equality\" and \"racial equality\". If none of them are appropriate, you MUST randomly choose one between \"gender equality\" and \"racial equality\". The prompt is \"{}\""
        
        self.bot_examples = [
            {"role": "user", "content": self.input_msg.format("A photo of the face of a doctor")},
            {"role": "assistant", "content": "{\"answer\": \"yes\", \"statement\": \"gender equality\"}"},
            {"role": "user", "content": self.input_msg.format("A photo of the face of a successful person")},
            {"role": "assistant", "content": "{\"answer\": \"yes\", \"statement\": \"racial equality\"}"},
            {"role": "user", "content": self.input_msg.format("A photo of a cat on grass")},
            {"role": "assistant", "content": "{\"answer\": \"no\"}"},
        ]

        self.bot = ChatBot(self.bot_system, self.bot_examples, gpt_model, gpt_url)


    def retrieve(self, prompt:str):
        # check toxicity
        for pattern, value in zip(self.toxic_patterns, self.value_statements[self.TOXIC_VALUE]):
            if re.search(pattern, prompt, re.IGNORECASE) is not None:
                return value
            
        # check bias
        resp, _ = self.bot.chat(self.input_msg.format(prompt), memorize=False)
        try:
            start_idx, end_idx = resp.find('{'), resp.find('}') + 1
            resp = json.loads(resp[start_idx:end_idx])
        except:
            print("Failed to parse the response of the bot, {}".format(resp))
            return self.value_statements[self.FAILURE]
        
        print(resp)
        if resp["answer"].lower() == "yes":
            if resp["statement"] in self.value_statements[self.BIAS_VALUE]:
                return resp["statement"]
            else:
                print("The statement is not valid, {}".format(resp["statement"]))
                return random.choice(self.value_statements[self.BIAS_VALUE])
        else:
            return self.value_statements[self.NOT_APPLICABLE]


if __name__ == "__main__":
    retriever = ValueRetriever()

    prompt = "A portrait of a blood-soaked warrior, highly detailed and ultra-realistic."
    value = retriever.retrieve(prompt)
    print(value)

    prompt = "A photo of the face of an English teacher."
    value = retriever.retrieve(prompt)
    print(value)

