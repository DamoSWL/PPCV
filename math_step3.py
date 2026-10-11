from transformers import AutoTokenizer, AutoModelForCausalLM


import numpy as np
import torch
import time
import re
from typing import Tuple, List
import json
# from grade_rstar import extract_math_answer, math_equiv
from grade_predictive import extract_answer,math_equal
import random
import torch.nn.functional as F
import torch
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import logging
import argparse
from vllm.inputs.data import TokenInputs   


MATH_COT_4_SHOT = """
Example 1:
The question is : Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?
The reasoning steps are:

The distance between two points $(x_1,y_1)$ and $(x_2,y_2)$ in the complex plane is given by the formula $\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$.
In this case, Joe's point is $(1,2)$ and Gracie's point is $(-1,1)$.
So the distance between their points is $\\sqrt{((-1)-(1))^2+((1)-(2))^2}=\\sqrt{(-2)^2+(-1)^2}=\\sqrt{4+1}=\\sqrt{5}$.
Therefore, Gracie and Joe's points are $\\boxed{\\sqrt{5}}$ units apart.
The answer is: \\boxed{\\sqrt{5}}


Example 2:
The question is :
Five socks, colored blue, brown, black, red, and purple are in a drawer. In how many different ways can we choose three socks from the drawer if the order of the socks does not matter?

The reasoning steps are:
This is a combination problem, since the order of the socks does not matter.
We want to choose 3 out of the 5 socks, so we can use the formula for combinations:
$\\binom{n}{k}=\\dfrac{n!}{k!(n-k)!}$.
In this case, $n=5$ (the total number of socks) and $k=3$ (the number of socks to choose).
Plugging in the values, we get $\\binom{5}{3}=\\dfrac{5!}{3!(5-3)!}=\\dfrac{5!}{3!2!}=\\dfrac{5\\times4\\times3\\times2\\times1}{3\\times2\\times1\\times2\\times1}=\\dfrac{5\\times4}{2\\times1}=\\boxed{10}$.
The answer is: \\boxed{10}

""".strip()




def load_vllm_model(model_path):
    llm = LLM(model=model_path,
              tensor_parallel_size=torch.cuda.device_count(),
              trust_remote_code=True,
              dtype = 'bfloat16',
            #   swap_space=32,
              max_model_len=16384,
              gpu_memory_utilization=0.7)

    return llm


def load_model(model_name, num_gpus=-1, start_id=0):
    # device = "cuda"
    # max_gpu_memory = 35
    # if device == "cuda":
    #     kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{model_name}/offload"}
    #     if num_gpus == -1:
    #         kwargs["device_map"] = "auto"
    #     else:
    #         num_gpus = int(num_gpus)
    #         if torch.cuda.device_count() != 1:
    #             kwargs.update({
    #                 "device_map": "auto",
    #                 "max_memory": {i: f"{max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
    #             })
    # elif device == "cpu":
    #     kwargs = {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name,
    #                                              low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    # if device == "cuda" and num_gpus == 1:
    #     model.cuda()
    return  tokenizer






@torch.inference_mode()
def generate_perturbed_question(llm,tokenizer,ori_question):
    
    while True:
        system_prompt = "You are a paraphrasing assistant specialized in math problems. Your task is to paraphrase a given math word problem. The core objective is to create a new, diverse expression of the problem while strictly preserving all numerical values, mathematical relationships, and the core logic required to solve it.\n"
        prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please create 5 paraphrased versions of the question below. Do NOT change any numbers or quantities. Maintain the exact mathematical operations and relationships. And vary the context and language. Please output paraphrased questions with serial number in square brackets and end with newline character.\n\n{ori_question}\n\n"},
                # {"role": "assistant", "content": ""}
                ]

        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        sampling_params = SamplingParams(n=1, temperature=0.85, top_p=0.9, max_tokens=2048, presence_penalty=0.6, frequency_penalty=0.3)
        outputs = llm.generate(prompts=prompt_ids, sampling_params=sampling_params, use_tqdm=False)

        questions = {'v0':ori_question}

        perturbed_questions = outputs[0].outputs[0].text.strip()
        print(perturbed_questions)
        print('===================================')
        pattern = r"\[\d+\]\s+"
        matches = list(re.finditer(pattern, perturbed_questions))
        matches_text = [m.group() for m in matches]
        matches_index = [m.start() for m in matches]

        if len(matches_index) > 0:
            break


    for i in range(len(matches_index)):
        if i != len(matches_index)-1:
            questions[f'v{i+1}'] = perturbed_questions[matches_index[i]:matches_index[i+1]]
        else:
            questions[f'v{i+1}'] = perturbed_questions[matches_index[i]:]

        questions[f'v{i+1}'] = questions[f'v{i+1}'][len(matches_text[i]):].strip()

        if i == 4:
            break

    
    return questions


@torch.inference_mode()
def generate_paths(llm, tokenizer, questions,max_new_tokens=3000,stop=None): 
    prompts = []
    for question in questions:
        system_prompt = "You are a specialized AI assistant designed to solve mathematical problems.\n"
        prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
                # {"role": "assistant", "content": ""}
                ]
        

        # system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format.\nI will give you some examples for reference.\n{MATH_COT_4_SHOT}"
        # prompt = [
        #         {"role": "system", "content": system_prompt},
        #         # {"role": "user", "content": f"The question: {question}\nPlease directly output the reasoning steps.\n"},
        #         {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
        #         # {"role": "assistant", "content": ""}
        #         ]


        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
       
    sampling_params = SamplingParams(n=1, temperature=0, top_p=1.0, max_tokens=max_new_tokens, stop=stop)
    outputs = llm.generate(prompts=prompt_ids, sampling_params=sampling_params,use_tqdm=False)

    texts = []
    for k, output in enumerate(outputs):
        text = []
        for j, output_2 in enumerate(output.outputs):
            text.append(output_2.text.strip())
        texts.append(text)
    return texts



@torch.inference_mode()
def obtain_part_ans_before_critical_token(llm, tokenizer, questions,ori_question_ans,max_new_tokens=3000,stop=None): 
    critical_token_info = defaultdict(list)

    ori_question_ans_ids = tokenizer.encode(ori_question_ans, add_special_tokens=False)
    for question in questions:

        system_prompt = "You are a specialized AI assistant designed to solve mathematical problems.\n"
        prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
                # {"role": "assistant", "content": ""}
                ]
        
        # system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format.\nI will give you some examples for reference.\n{MATH_COT_4_SHOT}"
        # prompt = [
        #         {"role": "system", "content": system_prompt},
        #         # {"role": "user", "content": f"The question: {question}\nPlease directly output the reasoning steps.\n"},
        #         {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
        #         # {"role": "assistant", "content": ""}
        #         ]


        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        question_ids = tokenizer(prompt, return_tensors="pt",add_special_tokens=False).input_ids
        final_prompt = prompt + ori_question_ans
        input_ids = tokenizer(final_prompt, return_tensors="pt",add_special_tokens=False).input_ids


        with torch.no_grad():
            outputs = llm(input_ids.cuda(),return_dict=True,output_attentions=True,output_hidden_states=True,use_cache=False)
            probs = F.softmax(outputs.logits, dim=-1).squeeze().to("cpu")

        top_probs,top_token_ids = probs.max(dim=-1)
        top_probs = top_probs[len(question_ids[0])-1:-1]
        top_token_ids = top_token_ids[len(question_ids[0])-1:-1]


        target_probs = torch.gather(probs, 1, input_ids[:,1:].to("cpu")).squeeze()[len(question_ids[0])-1:]
        target_token_ids = input_ids[:,len(question_ids[0]):].squeeze()

        candidate_critical_tokens_index = (top_token_ids != target_token_ids).nonzero(as_tuple=True)[0]

        for idx in candidate_critical_tokens_index:
            critical_token_info[idx.item()].append(top_probs[idx].item()-target_probs[idx].item())

        
    critical_token_index = 0
    critical_token_score = 0
    critical_token_count = 0

    for idx in critical_token_info:
        # score = sum(critical_token_info[idx])/len(critical_token_info[idx])
        score = max(critical_token_info[idx])
        if score > critical_token_score:
            critical_token_score = score
            critical_token_index = idx

    # for idx in critical_token_info:
    #     if len(critical_token_info[idx]) > critical_token_count:
    #         critical_token_count = len(critical_token_info[idx])
    #         critical_token_index = idx
    #         # critical_token_score = sum(critical_token_info[idx])/len(critical_token_info[idx])
    #         # critical_token_score = max(critical_token_info[idx])
    #         if critical_token_count > 1:
    #             critical_token_score = np.std(critical_token_info[idx])
    #     elif len(critical_token_info[idx]) == critical_token_count:
    #         # score = sum(critical_token_info[idx])/len(critical_token_info[idx])
    #         # score = max(critical_token_info[idx])
    #         score = 0
    #         if len(critical_token_info[idx]) > 1:
    #             score = np.std(critical_token_info[idx])
    #         if score > critical_token_score:
    #             critical_token_score = score
    #             critical_token_index = idx
    #     else:
    #         pass

    # critical_token_index = list(critical_token_info.keys())[0]

    return ori_question_ans_ids[:critical_token_index]



   
@torch.inference_mode()
def obtain_part_ans_with_critical_tokens(llm, tokenizer, question,ori_question_ans_token_ids,max_new_tokens=3000,stop=None,topk=10): 
    topk=topk
    sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=1, logprobs=topk, stop=stop)

    system_prompt = "You are a specialized AI assistant designed to solve mathematical problems.\n"
    prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
            # {"role": "assistant", "content": ""}
            ]

    # system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format.\nI will give you some examples for reference.\n{MATH_COT_4_SHOT}"
    # prompt = [
    #         {"role": "system", "content": system_prompt},
    #         # {"role": "user", "content": f"The question: {question}\nPlease directly output the reasoning steps.\n"},
    #         {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
    #         # {"role": "assistant", "content": ""}
    #         ]
    
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,enable_thinking=False)
    inputs = tokenizer.encode(prompt, add_special_tokens=False)
    inputs += ori_question_ans_token_ids

    vllm_inputs = TokenInputs()
    vllm_inputs['prompt_token_ids'] = inputs
    outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].logprobs[0]

    part_ans_critical_tokens = {}
    for i, token_id in enumerate(outputs.keys()):
        part_ans_critical_tokens[f'a{i}'] = tokenizer.decode(ori_question_ans_token_ids + [token_id],remove_special_tokens=True)


    return part_ans_critical_tokens

 

# def generate_path_paraphrased_question(llm, tokenizer, questions,part_ans,max_new_tokens=3000,stop=None):

#     results = {}
#     for idx, question in enumerate(questions):
#         sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=max_new_tokens, stop=stop)

#         system_prompt = "You are a specialized AI assistant designed to solve mathematical problems.\n"
#         prompt = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
#                 # {"role": "assistant", "content": ""}
#                 ]
        
#         # system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format.\nI will give you some examples for reference.\n{MATH_COT_4_SHOT}"
#         # prompt = [
#         #         {"role": "system", "content": system_prompt},
#         #         # {"role": "user", "content": f"The question: {question}\nPlease directly output the reasoning steps.\n"},
#         #         {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
#         #         # {"role": "assistant", "content": ""}
#         #         ]
    
#         prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,enable_thinking=False)
#         inputs = [prompt + part_ans[k] for k in part_ans]
#         inputs_ids = [tokenizer.encode(i, add_special_tokens=False) for i in inputs]

#         vllm_inputs = [TokenInputs() for _ in range(len(inputs_ids))]
#         for i in range(len(vllm_inputs)):
#             vllm_inputs[i]['prompt_token_ids'] = inputs_ids[i]
#         outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)


#         for k, output in enumerate(outputs):
#             if f'a{k}' not in results:
#                 results[f'a{k}'] = {}
#             for _, output_2 in enumerate(output.outputs):
#                 results[f'a{k}'][f'v{idx}'] = part_ans[f'a{k}'] + output_2.text


#     return results



def generate_path_paraphrased_question(llm, tokenizer, questions,part_ans,max_new_tokens=8192,stop=None,topk=10):

    results = {}
    all_vllm_inputs = []
    for idx, question in enumerate(questions):
        sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=max_new_tokens, stop=stop)

        system_prompt = "You are a specialized AI assistant designed to solve mathematical problems.\n"
        prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
                # {"role": "assistant", "content": ""}
                ]
        
        # system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format.\nI will give you some examples for reference.\n{MATH_COT_4_SHOT}"
        # prompt = [
        #         {"role": "system", "content": system_prompt},
        #         # {"role": "user", "content": f"The question: {question}\nPlease directly output the reasoning steps.\n"},
        #         {"role": "user", "content": f"Please solve the following problem step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\n"},
        #         # {"role": "assistant", "content": ""}
        #         ]
    
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,enable_thinking=False)
        inputs = [prompt + part_ans[k] for k in part_ans]
        inputs_ids = [tokenizer.encode(i, add_special_tokens=False) for i in inputs]

        vllm_inputs = [TokenInputs() for _ in range(len(inputs_ids))]
        for i in range(len(vllm_inputs)):
            vllm_inputs[i]['prompt_token_ids'] = inputs_ids[i]
        all_vllm_inputs.extend(vllm_inputs)


    outputs = llm.generate(prompts=all_vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

    for k, output in enumerate(outputs):
        if f'a{k%topk}' not in results:
            results[f'a{k%topk}'] = {}
        for _, output_2 in enumerate(output.outputs):
            results[f'a{k%topk}'][f'v{k//topk}'] = part_ans[f'a{k%topk}'] + output_2.text


    return results





def obtain_final_ans(questions,results,sim_model):
    tau = 2
    
    final_ans = None
    score_by_tokens = defaultdict(list)
    for k in results:
        flag, first_ans = extract_answer(results[k]['v0'])
        if not flag:
            first_ans = 'N/A'
        score_by_tokens[k] = [math_equal(extract_answer(res)[1],first_ans) for res in list(results[k].values())[1:]]

    scores = [sum(score_by_tokens[k]) for k in score_by_tokens]
    if not any(scores):
        final_ans = results['a0']['v0']
    else:
        # final_ans = results[f'a{scores.index(max(scores))}']['v0']

        if scores.count(max(scores)) == 1:
            final_ans = results[f'a{scores.index(max(scores))}']['v0']
        else:
            candidate_index = [k for k in range(len(scores)) if scores[k] == max(scores)]
            question_embedding = [sim_model.encode(questions[k], convert_to_tensor=True) for k in  questions]

            similarities = [util.cos_sim(question_embedding[0], question_embedding[k]).item() for k in range(len(question_embedding))][1:]
            similarities = torch.tensor(similarities).cuda()
            similarities = torch.exp(-tau*similarities)
            similarities = F.softmax(similarities,dim=-1).cpu()

            new_scores = [torch.sum(torch.tensor(score_by_tokens[f'a{k}'],dtype=torch.int)*similarities).item() for k in candidate_index]
            final_ans = results[f'a{candidate_index[new_scores.index(max(new_scores))]}']['v0']

        # candidate_index = [k for k in range(len(scores))]
        # question_embedding = [sim_model.encode(questions[k], convert_to_tensor=True) for k in  questions]

        # similarities = [util.cos_sim(question_embedding[0], question_embedding[k]).item() for k in range(len(question_embedding))][1:]
        # similarities = torch.tensor(similarities).cuda()
        # similarities = torch.exp(-tau*similarities)
        # similarities = F.softmax(similarities,dim=-1).cpu()

        # new_scores = [torch.sum(torch.tensor(score_by_tokens[f'a{k}'],dtype=torch.int)*similarities).item() for k in candidate_index]
        # final_ans = results[f'a{candidate_index[new_scores.index(max(new_scores))]}']['v0']

    return final_ans


def set_seed(seed):
    # Python built-in
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU (single GPU)
    torch.cuda.manual_seed(seed)
    
    # PyTorch GPU (all GPUs)
    torch.cuda.manual_seed_all(seed)
    
    # Ensures deterministic behavior in cuDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For DataLoader with num_workers > 0
    torch.Generator().manual_seed(seed)

if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="PPCV")
    parser.add_argument('--dataset', type=str, default='math500',
                        help='Dataset type')
    parser.add_argument('--topk', type=int, default=10,
                        help='Dataset type')
    parser.add_argument('--seed', type=int, default=10,
                        help='seed')
    parser.add_argument('--model', type=str, default='Qwen3-32B',
                        help='seed')
    
    args = parser.parse_args()

    logging.basicConfig(filemode='a',filename=f'logg_{args.dataset}.txt', level=logging.DEBUG,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    set_seed(args.seed)

    # model_name = "Llama-3.1-8B-Instruct"
    # model_name = "DeepSeek-R1-Distill-Llama-70B"
    # model_name = "Qwen3-32B"
    # model_name = "Qwen3-30B-A3B-Instruct-2507"
    model_name = args.model

    llm_vllm = load_vllm_model(model_name)
    tokenizer = load_model(model_name)

    sim_model = SentenceTransformer('all-MiniLM-L6-v2').cuda().eval()

    saved_file_name = f'qwen3_result_{args.dataset}_top{args.topk}_seed{args.seed}.json'
    with open(f'/project/sds-rise/weili/Project_2/LLM_eval/{saved_file_name}') as f:
        test_data = json.load(f)
    

    match_flag = []
    saved_results = []

    start_time = time.time()
    culmative_time = 0

    for i in range(len(test_data)):
        logging.info(f'Processing sample {i}...')
        # ori_question = test_data[i]['question']
        gt_ans = test_data[i]['answer']
        questions = test_data[i]['question']
        ori_question_ans = test_data[i]['ori_question_ans']
        part_ans = test_data[i]['part_ans']

        start_time = time.time()
        

        # questions = generate_perturbed_question(llm_vllm,tokenizer,ori_question)
        # ori_question_ans = generate_paths(llm_vllm,tokenizer,[questions['v0']])[0][0]

    

        # part_ori_question_ans_token_ids = obtain_part_ans_before_critical_token(llm_huggingface, tokenizer, list(questions.values())[1:], ori_question_ans) 

        # # logging.info('===================================')
        # # logging.info(tokenizer.decode(part_ori_question_ans_token_ids,remove_special_tokens=True))
        # # logging.info(part_ori_question_ans_token_ids)
        # logging.info('===================================')

        part_ori_question_ans_token_ids = tokenizer.encode(part_ans, add_special_tokens=False)
        part_ori_question_ans_critical_tokens = obtain_part_ans_with_critical_tokens(llm_vllm,tokenizer,list(questions.values())[0],part_ori_question_ans_token_ids,topk=args.topk)


        # logging.info('===================================')
        # logging.info(part_ori_question_ans_critical_tokens)
        # logging.info('===================================')


        results = generate_path_paraphrased_question(llm_vllm,tokenizer,list(questions.values()),part_ori_question_ans_critical_tokens,topk=args.topk)

        culmative_time += time.time() - start_time
        # logging.info('+++++++++++++++++++++++++++++++++++++')
        # # logging.info(results['a0']['v0'])
        # # exit()

        final_result = obtain_final_ans(questions,results,sim_model)

        match_flag.append(math_equal(extract_answer(final_result)[1],gt_ans))

        saved_result = {}
        saved_result['index'] = i
        saved_result['question'] = questions
        saved_result['ori_question_ans'] = ori_question_ans
        saved_result['part_ans'] = part_ans
        saved_result['part_ans_critical_token'] = part_ori_question_ans_critical_tokens
        saved_result['candidate_ans'] = results
        saved_result['predictive_ans'] = final_result
        saved_result['gt_ans'] = gt_ans
        saved_result['match_flag'] = match_flag[i]
        saved_results.append(saved_result)

        with open(saved_file_name, 'w') as f:
            json.dump(saved_results, f, indent=2)

    print(f"Match Accuracy: {sum(match_flag) / len(match_flag) * 100:.2f}%")
    saved_results.append({"Match Accuracy": sum(match_flag) / len(match_flag) * 100.0})

    # print(f'eplapsed time {culmative_time}')
    logging.info(f'culmative_time {culmative_time}')


    # with open('result_math_eval_max_new_6.json', 'w') as f:
    #     json.dump(saved_results, f, indent=2)

    with open(saved_file_name, 'w') as f:
        json.dump(saved_results, f, indent=2)
