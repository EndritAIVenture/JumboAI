# 1.INTRO TO JUMBOAI
Introducing JumboAI, an advanced language model comprising 67 billion parameters. It has been trained from scratch on a vast dataset of 2 trillion tokens in both English and Chinese. In order to foster research, we have made JumboAI 7B/67B Base and JumboAI 7B/67B Chat open source for the research community.
![image](https://github.com/user-attachments/assets/d20657e8-3450-4df9-91d3-b7cafedc42e9)

-Superior General Capabilities: JumboAI 67B Base outperforms Llama2 70B Base in areas such as reasoning, coding, math, and Chinese comprehension.

-Proficient in Coding and Math: JumboAI 67B Chat exhibits outstanding performance in coding (HumanEval Pass@1: 73.78) and mathematics (GSM8K 0-shot: 84.1, Math 0-shot: 32.6). It also demonstrates remarkable generalization abilities, as evidenced by its exceptional score of 65 on the Hungarian National High School Exam.

-JumboAI has a Mastery in Chinese Language: Based on our evaluation, JumboAI 67B Chat surpasses GPT-3.5 in Chinese.


# 2. EVALUATION RESULTS

Base Model
We evaluate our models and some baseline models on a series of representative benchmarks in English and Chinese. More results can be found in the evaluation folder. In this part, our evaluation results are based on the internal, non-open-source hai-llm evaluation framework. Please note that there may be slight discrepancies when using the converted HuggingFace models.

model|HellaSwag	|Trivia QA|	MMLU	|GSM8K|	Human Eval|	BBH	|CEval|	CMMLU|	ChineseQA

LLaMA-2-7B	75.6	63.8	45.8	15.5	14.6	38.5	33.9	32.6	21.5
