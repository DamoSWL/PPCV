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
import argparse
import logging
from vllm.inputs.data import TokenInputs  

paraphrase_prompt = """You are given a multiple-choice question from a reasoning benchmark. 
Your task is to generate multiple different paraphrased versions of the question. 
Each version must preserve the original meaning, answer choices, and scientific/commonsense accuracy.

Guidelines:
- Keep the question logically equivalent to the original.
- Do not change the correct answer.
- Do not add or remove details that would alter the difficulty.
- Use different phrasing, sentence structures, or synonyms for each version.
- Keep the multiple-choice options (A, B, C, D, etc.) unchanged.


Example Input:
A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?
(A) antisense
(B) lariat
(C) R-loops
(D) polyA tail

Example Output:
[1]
A gene contains many exons, with the central ones responsible for forming triple-helical repeats that link the cytoskeleton to the sarcolemma and the extracellular matrix. Each exon typically encodes one folded triple alpha helix. The most frequent mutations are deletions of central exons, producing out-of-frame peptides and degenerative disease. A possible therapy is to introduce a Morpholino that binds to the 5′ end of the defective exon in pre-mRNA. This blocks the spliceosome, skips the exon, and restores the reading frame. Since some exon losses are tolerated, which of the following is NOT part of this treatment?
(A) antisense
(B) lariat
(C) R-loops
(D) polyA tail

[2]
Within a large gene, the middle exons encode triple-helical domains that link the cytoskeleton with the sarcolemma and the extracellular environment. Each exon generally codes for one folded triple helix. The most common genetic errors involve deleting central exons, leading to out-of-frame products and progressive tissue degeneration. A treatment strategy uses a Morpholino that targets the 5′ end of the faulty exon in pre-mRNA, preventing spliceosome attachment. This results in exon skipping and in-frame splicing. Given that several missing exons are tolerated, which element is NOT required for this therapeutic method?
(A) antisense
(B) lariat
(C) R-loops
(D) polyA tail

[3]
A large gene consists of many exons, with the central ones producing folded triple helices that anchor the cytoskeleton to the sarcolemma and extracellular structures. Normally, one exon corresponds to one folded triple alpha helix. The most frequent mutations involve deleting these central exons, creating out-of-frame peptides that drive degenerative disease. Treatment can include a Morpholino that binds to the 5′ end of the affected exon in pre-mRNA, blocking the spliceosome and promoting exon skipping with in-frame repair. Because several exon deletions are tolerated, which of the following structures is NOT part of this therapeutic approach?
(A) antisense
(B) lariat
(C) R-loops
(D) polyA tail"""




def load_vllm_model(model_path):
    llm = LLM(model=model_path,
              tensor_parallel_size=torch.cuda.device_count(),
              trust_remote_code=True,
              dtype = 'bfloat16',
            #   swap_space=32,
              max_model_len=32768,
              gpu_memory_utilization=0.3)

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
                                                #  low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    # if device == "cuda" and num_gpus == 1:
    #     model.cuda()
    # return model, tokenizer
    return tokenizer




@torch.inference_mode()
def generate_perturbed_question(llm,tokenizer,ori_question):
    
    while True:
        system_prompt = paraphrase_prompt
        prompt = [
                {"role": "system", "content": system_prompt},
                # {"role": "user", "content": f"Please create 3 paraphrased versions of the question below. Only paraphrases the question and leaves the multiple-choice options untouched. Please output paraphrased questions starting with a bracketed serial number and ending with a new line\n\n{ori_question}\n\n"},
                {"role": "user", "content": f"Please create 3 paraphrased versions of the question below. Only paraphrases the question and leaves the multiple-choice options untouched. Please output paraphrased questions starting with a bracketed serial number like [1] [2] and ending with a new line\n\n{ori_question}\n\n"},
                # {"role": "assistant", "content": ""}
                ]

        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        sampling_params = SamplingParams(n=1, temperature=0.85, top_p=0.9, max_tokens=2048, presence_penalty=0.6, frequency_penalty=0.3)
        vllm_inputs = TokenInputs()
        vllm_inputs['prompt_token_ids'] = prompt_ids
        outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

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

        if i == 2:
            break

    
    return questions


@torch.inference_mode()
def generate_paths(llm, tokenizer, questions,max_new_tokens=2048,stop=None): 
    prompts = []
    for question in questions:
        system_prompt = "You are a specialized AI assistant designed to solve complex reasoning problems.\n"
        prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Your task is to carefully read the following question, apply logical reasoning, scientific knowledge, and commonsense inference, and arrive at the correct answer. Always explain your reasoning step by step in clear language. At the end of your response, output the final choice in the following format:\n\nThe answer is: (<option>)\n\nwhere <option> is the correct option letter (A, B, C, D, etc.), enclosed in parentheses.\n\n{question}\n\n"},
                # {"role": "user", "content": f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response and enclose the correct option with parentheses.\n\n{question}\n\n"},
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
    vllm_inputs = [TokenInputs() for _ in range(len(prompt_ids))]
    for i in range(len(vllm_inputs)):
        vllm_inputs[i]['prompt_token_ids'] = prompt_ids[i]
    outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params,use_tqdm=False)


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
def obtain_part_ans_with_critical_tokens(llm, tokenizer, question,ori_question_ans_token_ids,max_new_tokens=3000,stop=None): 
    topk=10
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
    
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False)
    inputs += ori_question_ans_token_ids

    outputs = llm.generate(prompt_token_ids=inputs, sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].logprobs[0]

    part_ans_critical_tokens = {}
    for i, token_id in enumerate(outputs.keys()):
        part_ans_critical_tokens[f'a{i}'] = tokenizer.decode(ori_question_ans_token_ids + [token_id],remove_special_tokens=True)


    return part_ans_critical_tokens

 

def generate_path_paraphrased_question(llm, tokenizer, questions,part_ans,max_new_tokens=3000,stop=None):

    results = {}
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
    
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = [prompt + part_ans[k] for k in part_ans]
        inputs_ids = [tokenizer.encode(i, add_special_tokens=False) for i in inputs]


        outputs = llm.generate(prompt_token_ids=inputs_ids, sampling_params=sampling_params, use_tqdm=False)


        for k, output in enumerate(outputs):
            if f'a{k}' not in results:
                results[f'a{k}'] = {}
            for _, output_2 in enumerate(output.outputs):
                results[f'a{k}'][f'v{idx}'] = part_ans[f'a{k}'] + output_2.text


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

    parser = argparse.ArgumentParser(description="Phi-Decoding Algorithm")
    parser.add_argument('--dataset', type=str, default='gsm',
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
    # sim_model = SentenceTransformer('all-MiniLM-L6-v2').cuda().eval()
    model_name = args.model

    llm_vllm = load_vllm_model(model_name)
    tokenizer = load_model(model_name)

    saved_file_name = f'result_{args.dataset}.json'
    with open(f'/project/sds-rise/weili/Project_2/LLM_eval/datasets/{args.dataset}_test.json') as f:
        test_data = json.load(f)
    
    match_flag = []
    saved_results = []

    para_time = 0
    init_ans_time = 0


    for i in range(len(test_data)):
        # if i+1>=199:
        #     break
        
        ori_question = test_data[i]['question']
        gt_ans = test_data[i]['answer']

        start_time = time.time()
        questions = generate_perturbed_question(llm_vllm,tokenizer,ori_question)
        para_time += time.time() - start_time
        start_time = time.time()
        ori_question_ans = generate_paths(llm_vllm,tokenizer,[questions['v0']])[0][0]
        init_ans_time += time.time() - start_time

        saved_result = {}
        saved_result['index'] = i
        saved_result['question'] = questions
        saved_result['ori_question_ans'] = ori_question_ans
        saved_result['answer'] = gt_ans
        

        saved_results.append(saved_result)


        with open(saved_file_name, 'w') as f:
            json.dump(saved_results, f, indent=2)



        # ori_question_ans = generate_paths(llm_vllm,tokenizer,[questions['v0']])[0][0]

    

        # part_ori_question_ans_token_ids = obtain_part_ans_before_critical_token(llm_huggingface, tokenizer, list(questions.values())[1:], ori_question_ans) 

        # # logging.info('===================================')
        # # logging.info(tokenizer.decode(part_ori_question_ans_token_ids,remove_special_tokens=True))
        # # logging.info(part_ori_question_ans_token_ids)
        # # logging.info('===================================')


        # part_ori_question_ans_critical_tokens = obtain_part_ans_with_critical_tokens(llm_vllm,tokenizer,ori_question,part_ori_question_ans_token_ids)


        # # logging.info('===================================')
        # # logging.info(part_ori_question_ans_critical_tokens)
        # # logging.info('===================================')


        # results = generate_path_paraphrased_question(llm_vllm,tokenizer,list(questions.values()),part_ori_question_ans_critical_tokens)

        # # logging.info(results['a0']['v0'])
        # # exit()

        # final_result = obtain_final_ans(questions,results,sim_model)

        # match_flag.append(math_equal(extract_answer(final_result)[1],gt_ans))

        # saved_result = {}
        # saved_result['index'] = i
        # saved_result['question'] = questions
        # saved_result['ori_question_ans'] = ori_question_ans
        # saved_result['part_ans'] = tokenizer.decode(part_ori_question_ans_token_ids,remove_special_tokens=True)
        # saved_result['part_ans_critical_token'] = part_ori_question_ans_critical_tokens
        # saved_result['candidate_ans'] = results
        # saved_result['predictive_ans'] = final_result
        # saved_result['gt_ans'] = gt_ans
        # saved_result['match_flag'] = match_flag[i]
        # saved_results.append(saved_result)

        # with open('result_math_eval_max_new_6.json', 'w') as f:
        #     json.dump(saved_results, f, indent=2)

    # print(f"Match Accuracy: {sum(match_flag) / len(match_flag) * 100:.2f}%")
    # saved_results.append({"Match Accuracy": sum(match_flag) / len(match_flag) * 100.0})


    # with open('result_math_eval_max_new_6.json', 'w') as f:
    #     json.dump(saved_results, f, indent=2)

    # print(f'eplapsed time {time.time()-start_time}')
    logging.info(f'para_time {para_time}')
    logging.info(f'init_ans_time {init_ans_time}')


    with open(saved_file_name, 'w') as f:
        json.dump(saved_results, f, indent=2)
