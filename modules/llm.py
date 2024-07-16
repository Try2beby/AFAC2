import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from enum import Enum
import os

from google.colab import drive

drive.mount("/content/drive")
# os.chdir("/content/drive/MyDrive/LLM")

os.environ["HF_HOME"] = "/content/drive/MyDrive/cache"


device = "cuda"  # the device to load the model onto

extra_kwargs = {}
extra_kwargs["model"] = {}
extra_kwargs["model"]["torch_dtype"] = torch.float16
extra_kwargs["model"]["low_cpu_mem_usage"] = True
extra_kwargs["model"]["device_map"] = {"": device}

# 定义提示模板
prompt_template = """使用以下上下文内容来回答后面的问题。如果你不知道答案，就回答你不知道，不要试图编造答案，也不要加入多余的信息。

{context}

问题：{question}
用中文回答：
"""


class MODEL(str, Enum):
    internlm2_5_7b = "internlm/internlm2_5-7b"
    glm_4_9b = "THUDM/glm-4-9b"


def load_model(model: MODEL = MODEL.glm_4_9b.value, extra_kwargs=extra_kwargs):
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
    model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=True, **extra_kwargs["model"]
    )
    return model, tokenizer


def get_response(model: MODEL, tokenizer, context, question):
    model, tokenizer = load_model(model)
    prompt = prompt_template.format(context=context, question=question)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(inputs.input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
