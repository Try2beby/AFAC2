import json
import os
import concurrent.futures
from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# 定义文件名和路径
file_name = dict(dev="dev.json", test="test.json", train="train.json")

data_path = "./round1_training_data/"
result_path = "./result/"

# 读取数据
data = dict()
for key in file_name:
    with open(data_path + file_name[key], "r") as f:
        data[key] = json.load(f)

# 初始化ChatZhipuAI
llm = ChatZhipuAI(
    temperature=0,
    zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    model_name="GLM-4-0520",
    # model_name="GLM-4-Flash",
)

# 定义提示模板
prompt_template = """使用以下上下文内容来回答后面的问题。如果你不知道答案，就回答你不知道，不要试图编造答案，也不要加入多余的信息。

{context}

问题：{question}
用中文回答：
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

context_template = """
产品名：{product_name}
条款：{clause}
"""


# 定义获取查询上下文的函数
def get_query_context(split="train", index=0):
    question = data[split][index]["问题"]
    context = context_template.format(
        product_name=data[split][index]["产品名"], clause=data[split][index]["条款"]
    )
    return dict(question=question, context=context)


# 定义并行处理函数
def process_query(index, split="train"):
    query_context = get_query_context(split, index)
    response = llm.invoke(PROMPT.format(**query_context)).content
    return index, response


# 并行调用的主函数
def parallel_process_queries(split="train", num_queries=5, max_workers=5):
    results_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_query, i, split) for i in range(num_queries)]
        for future in concurrent.futures.as_completed(futures):
            index, result = future.result()
            results_dict[index] = result

    # 将结果写回到原数据字典中
    for index, result in results_dict.items():
        data[split][index]["回答"] = result

    # 保存修改后的数据到文件
    with open(data_path + file_name[split], "w") as f:
        json.dump(data[split], f, ensure_ascii=False, indent=4)

    return results_dict


# 使用示例
if __name__ == "__main__":
    split = "test"
    num_queries = 5  # 指定并行处理的查询数量
    max_workers = 1  # 指定并行数
    test = "使用以下上下文内容来回答后面的问题。如果你不知道答案，就回答你不知道，不要试图编造答案，也不要加入多余的信息。\n\n\n产品名：太享E保百万医疗险\n条款：合同构成 本合同是您与我们约定保险权利义务关系的协议，包括本保险条款、保险单及其他保险凭证、投保单及其他投保文件、合法有效的声明、批注、批单及其他您与我们共同认可的书面协议。\n\n\n问题：在太享E保百万医疗险合同中，保险条款的作用是什么？\n用中文回答：\n"
    print(llm.invoke(test).content)
    # results = parallel_process_queries(split, num_queries, max_workers)
