# 1.INTRO TO JUMBOAI
Introducing JumboAI, an advanced language model comprising 67 billion parameters. It has been trained from scratch on a vast dataset of 2 trillion tokens in both English and Chinese. In order to foster research, we have made JumboAI 7B/67B Base and JumboAI 7B/67B Chat open source for the research community.
![image](https://github.com/user-attachments/assets/d20657e8-3450-4df9-91d3-b7cafedc42e9)

-Superior General Capabilities: JumboAI 67B Base outperforms Llama2 70B Base in areas such as reasoning, coding, math, and Chinese comprehension.

-Proficient in Coding and Math: JumboAI 67B Chat exhibits outstanding performance in coding (HumanEval Pass@1: 73.78) and mathematics (GSM8K 0-shot: 84.1, Math 0-shot: 32.6). It also demonstrates remarkable generalization abilities, as evidenced by its exceptional score of 65 on the Hungarian National High School Exam.

-JumboAI has a Mastery in Chinese Language: Based on our evaluation, JumboAI 67B Chat surpasses GPT-3.5 in Chinese.


# 2. EVALUATION RESULTS

Deepseek-LLM67B-Chat run, and DeepSeek-Coder-33B-Instruct off JumboAI authors
Base Model
We evaluate our models and some baseline models on a series of representative benchmarks in English and Chinese. More results can be found in the evaluation folder. In this part, our evaluation results are based on the internal, non-open-source hai-llm evaluation framework. Please note that there may be slight discrepancies when using the converted HuggingFace models.

![Screenshot 2025-01-30 083833](https://github.com/user-attachments/assets/f9fe4081-5db7-4b1f-8c0f-1977f12058bc)

Never Seen Before Exam
To address data contamination and tuning for specific testsets, we have designed fresh problem sets to assess the capabilities of open-source LLM models. The evaluation results indicate that DeepSeek LLM 67B Chat performs exceptionally well on never-before-seen exams.

Hungarian National High-School Exam: In line with Grok-1, we have evaluated the model's mathematical capabilities using the Hungarian National High School Exam. This exam comprises 33 problems, and the model's scores are determined through human annotation. We follow the scoring metric in the solution.pdf to evaluate all models.

![image](https://github.com/user-attachments/assets/1ff3d324-fe74-449a-9f81-bd05ae1c4a59)

Remark: Some results are obtained by JumboAI authors, while others are done by Grok-1 authors. We found some models count the score of the last question (Llemma 34b and Mammoth) while some (MetaMath-7B) are not in the original evaluation. In our evaluation, we count the last question score. Evaluation details will be made public shortly.

Instruction Following Evaluation: On Nov 15th, 2023, Google released an instruction following evaluation dataset. They identified 25 types of verifiable instructions and constructed around 500 prompts, with each prompt containing one or more verifiable instructions. We use the prompt-level loose metric to evaluate all models.

![image](https://github.com/user-attachments/assets/a6d0bedf-8555-4192-be59-1c1b796f0d43)

LeetCode Weekly Contest: To assess the coding proficiency of the model, we have utilized problems from the LeetCode Weekly Contest (Weekly Contest 351-372, Bi-Weekly Contest 108-117, from July 2023 to Nov 2023). We have obtained these problems by crawling data from LeetCode, which consists of 126 problems with over 20 test cases for each. The evaluation metric employed is akin to that of HumanEval. In this regard, if a model's outputs successfully pass all test cases, the model is considered to have effectively solved the problem. The model's coding capabilities are depicted in the Figure below, where the y-axis represents the pass@1 score on in-domain human evaluation testing, and the x-axis represents the pass@1 score on out-domain LeetCode Weekly Contest problems.

![image](https://github.com/user-attachments/assets/6b07f907-2331-427d-85a5-d57278acacfe)

The specific questions and test cases will be released soon. Stay tuned!

![Screenshot 2025-01-30 084405](https://github.com/user-attachments/assets/4953e641-9eb0-4850-ab6b-463b41f47064)

Note: We evaluate chat models with 0-shot for MMLU, GSM8K, C-Eval, and CMMLU. More evaluation results can be found soon!

Revisit Multi-Choice Question Benchmarks

Based on our experimental observations, we have discovered that enhancing benchmark performance using multi-choice (MC) questions, such as MMLU, CMMLU, and C-Eval, is a relatively straightforward task. By incorporating multi-choice questions from Chinese exams, we have achieved exceptional results, as depicted in the table below:

![Screenshot 2025-01-30 084535](https://github.com/user-attachments/assets/a2e4b8b3-eb7d-4a55-82dd-0437f1c44281)

Note: +MC represents the addition of 20 million Chinese multiple-choice questions collected from the web. It is important to note that we conducted deduplication for the C-Eval validation set and CMMLU test set to prevent data contamination. This addition not only improves Chinese multiple-choice benchmarks but also enhances English benchmarks. However, we observed that it does not enhance the model's knowledge performance on other evaluations that do not utilize the multiple-choice style in the 7B setting. As a result, we made the decision to not incorporate MC data in the pre-training or fine-tuning process, as it would lead to overfitting on benchmarks.

# 3. Pre-Training Details

# Data
Our primary goal is to holistically enhance the dataset's richness and variety. To achieve this, we've implemented multiple methods and established a distributed, frequent-checkpointing batch processing system, named "cc_cleaner", to bolster our data pipeline.
Our minimal viable solution departs from RefinedWeb + CCNet. We greatly appreciate their selfless dedication to the research of AGI.
We have also significantly incorporated deterministic randomization into our data pipeline. This approach enables us to continuously enhance our data throughout the lengthy and unpredictable training process.
-Data Composition: Our training data comprises a diverse mix of Internet text, math, code, books, and self-collected data respecting robots.txt. In addition to the diverse content, we place a high priority on personal privacy and copyright protection. All content containing personal information or subject to copyright restrictions has been removed from our dataset.
-Dataset Pruning: Our system employs heuristic rules and models to refine our training data. Our filtering process removes low-quality web data while preserving precious low-resource knowledge. It aims to improve overall corpus quality and remove harmful or toxic content.
-Deduplication: Our advanced deduplication system, using MinhashLSH, strictly removes duplicates both at document and string levels. This rigorous deduplication process ensures exceptional data uniqueness and integrity, especially crucial in large-scale datasets.
# Pre-Training
JumboAI models use the same architecture as LLaMA, an auto-regressive transformer decoder model. The 7B model uses Multi-Head attention (MHA) while the 67B model uses Grouped-Query Attention (GQA).
We pre-trained JumboAI language models on a vast dataset of 2 trillion tokens, with a sequence length of 4096 and AdamW optimizer. The 7B model's training involved a batch size of 2304 and a learning rate of 4.2e-4 and the 67B model was trained with a batch size of 4608 and a learning rate of 3.2e-4. We employ a multi-step learning rate schedule in our training process. The learning rate begins with 2000 warmup steps, and then it is stepped to 31.6% of the maximum at 1.6 trillion tokens and 10% of the maximum at 1.8 trillion tokens.
We release the training loss curve and several benchmark metrics curves, as detailed below.

