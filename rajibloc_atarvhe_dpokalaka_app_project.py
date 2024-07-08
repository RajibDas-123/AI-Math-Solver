import pandas as pd
from tqdm import tqdm
import time
import torch
import gc
import re
import sys
import subprocess
import math
import random
import  kagglehub
import streamlit as st
from collections import defaultdict

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    StoppingCriteria,
    set_seed
)
from transformers import BitsAndBytesConfig
import transformers

torch.backends.cuda.enable_mem_efficient_sdp(False)
set_seed(42)



def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)




def return_last_print(output, n):
    lines = output.strip().split('\n')
    if lines:
        return lines[n]
    else:
        return ""

def process_code(code, return_shell_output=False):
    
    def repl(match):
        if "real" not in match.group():
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')
    code = re.sub(r"symbols\([^)]+\)", repl, code)

    if return_shell_output:
        code = code.replace('\n', '\n    ')
            # Add a try...except block
        code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)
    
    if not return_shell_output:
        print(code)
    with open('code.py', 'w') as fout:
        fout.write(code)
    
    batcmd = 'timeout 7 ' + sys.executable + ' code.py'
    try:
        shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        return_value = return_last_print(shell_output, -1)
        print(shell_output)
        if return_shell_output:
            if return_value=='FAIL':
                CODE_STATUS = False
                return_value = return_last_print(shell_output, -2)
                if "not defined" in return_value:
                    return_value+='\nTry checking the formatting and imports'
            else:
                CODE_STATUS = True
            return return_value, CODE_STATUS  
        code_output = round(float(eval(return_value))) % 1000
    except Exception as e:
        print(e,'shell_output')
        code_output = -1
    
    if return_shell_output:
        if code_output==-1:
            CODE_STATUS = False
        else:
            CODE_STATUS = True
        return code_output, CODE_STATUS  
    
    return code_output


def process_text_output(output):
    result = output    
    try:
        result_output = re.findall(r'\\boxed\{(\d+)\}', result)

        print('BOXED', result_output)
        if not len(result_output):
            result_output = naive_parse(result)
        else:
            result_output = result_output[-1]

        print('BOXED FINAL', result_output)
        if not len(result_output):
            result_output = -1
        
        else:
            result_output = round(float(eval(result_output))) % 1000
    
    except Exception as e:
        print(e)
        print('ERROR PARSING TEXT')
        result_output = -1
    
    return result_output


torch.cuda.empty_cache()
gc.collect()


n_repetitions = 1
TOTAL_TOKENS = 2048
TIME_LIMIT = 31500

MODEL_PATH = '/mnt/home/pandavis/.cache/kagglehub/datasets/olyatsimboy/deepseek-math/versions/1'

#MODEL_PATH = "/deepseek-math"
DEEP = True

config = AutoConfig.from_pretrained(MODEL_PATH)
config.gradient_checkpointing = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True, 
    config=config
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype='auto',
)

from transformers import StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            last_token = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop,last_token)):
                return True
        return False


stop_words = ["```output", "```python", "```\nOutput" , ")\n```" , "``````output"] #,  
stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])




code = """Below is a math problem you are to solve (positive numerical answer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even a 5th grade student can follow your instructions, and remember, your final answer should be positive integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{}.

Approach:"""


cot = """Below is a math problem you are to solve (positive numerical answer!):
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{}.\n\n"""

promplt_options = [code,cot]



#df_train = pd.read_csv('Data/AIMO/data.csv')
#df_train.head()


import re
from collections import defaultdict
from collections import Counter

from numpy.random import choice
import numpy as np

tool_instruction = '\n\nPlease integrate natural language reasoning with programs to solve the above problem, and put your final numerical answer within \\boxed{}.\nNote that the intermediary calculations may be real numbers, but the final numercal answer would always be an integer.'

temperature = 0.9
top_p = 1.0

temperature_coding = 0.9
top_p_coding = 1.0

   
total_results = {}
total_answers = {}
best_stats = {}
total_outputs = {}
question_type_counts = {}
starting_counts = (2,3)
best_jj = -1
best_solution = ""



def solve_math_question(question_input):
    all_outputs = []
    try:
        i = 0
        
        for jj in tqdm(range(n_repetitions)):   
            problem = question_input
            best, best_count = best_stats.get(i,(-1,-1))
            if best_count>np.sqrt(jj):
                yield f"Skipping because already found best answer for repetition {jj}\n"
                continue

            outputs = total_outputs.get(i,[])
            text_answers, code_answers = question_type_counts.get(i,starting_counts)
            results = total_results.get(i,[])
            answers = total_answers.get(i,[])

            for _ in range(5):
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.2)

            try:
                ALREADY_GEN = 0
                code_error = None
                code_error_count = 0
                code_output = -1

                counts = np.array([text_answers,code_answers])

                draw = choice(promplt_options, 1,
                              p=counts/counts.sum())

                initail_message = draw[0].format(problem,"{}")            
                prompt = f"User: {initail_message}"

                current_printed = len(prompt)
                yield f"{jj}_{prompt}\n"

                model_inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                input_len = len(model_inputs['input_ids'][0])

                generation_output = model.generate(**model_inputs, 
                                                   max_new_tokens=TOTAL_TOKENS-ALREADY_GEN,
                                                   return_dict_in_generate=True,
                                                   do_sample = True,
                                                   temperature = temperature,
                                                   top_p = top_p,
                                                   num_return_sequences=1, stopping_criteria = stopping_criteria)

                output_ids = generation_output.sequences[0]

                decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)
                
                yield f"{decoded_output[current_printed:]}\n"
                current_printed += len(decoded_output[current_printed:])
                cummulative_code = ""


                stop_word_cond = False
                for stop_word in stop_words:
                    stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):]==stop_word)


                while (stop_word_cond) and (ALREADY_GEN<(TOTAL_TOKENS)):

                    if (decoded_output[-len("```python"):]=="```python"):
                        temperature_inner=temperature_coding
                        top_p_inner = top_p_coding
                        prompt = decoded_output
                    else:
                        temperature_inner=temperature
                        top_p_inner = top_p
                        try:
                            if (decoded_output[-len("``````output"):]=="``````output"):
                                code_text = decoded_output.split('```python')[-1].split("``````")[0]
                            else:
                                code_text = decoded_output.split('```python')[-1].split("```")[0]


                            cummulative_code+=code_text
                            code_output, CODE_STATUS = process_code(cummulative_code, return_shell_output=True)
                            yield f'CODE RESULTS: {code_output}\n'

                            if code_error==code_output:
                                code_error_count+=1
                            else:
                                code_error=code_output
                                code_error_count = 0

                            if not CODE_STATUS:
                                cummulative_code = cummulative_code[:-len(code_text)]

                                if code_error_count>=1:
                                    yield "REPEATED ERRORS\n"
                                    break

                        except Exception as e:
                            yield f"{e}\nERROR PARSING CODE\n"
                            code_output = -1

                        if code_output!=-1:
                            if (decoded_output[-len(")\n```"):]==")\n```"):
                                prompt = decoded_output+'```output\n'+str(code_output)+'\n```\n'
                            else:
                                prompt = decoded_output+'\n'+str(code_output)+'\n```\n'
                        else:
                            prompt = decoded_output
                            cummulative_code=""


                    model_inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                    ALREADY_GEN =  len(model_inputs['input_ids'][0])-input_len

                    old_values = generation_output.past_key_values

                    generation_output = model.generate(**model_inputs, 
                                                       max_new_tokens=TOTAL_TOKENS-ALREADY_GEN, 
                                                       return_dict_in_generate=True,
                                                       past_key_values=old_values,
                                                       do_sample = True,
                                                       temperature = temperature_inner,
                                                       top_p = top_p_inner,
                                                       num_return_sequences=1, stopping_criteria = stopping_criteria)

                    output_ids = generation_output.sequences[0]

                    decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)
                    yield f"\nINTERMEDIATE OUTPUT:\n{decoded_output[current_printed:]}\n"
                    current_printed+=len(decoded_output[current_printed:])

                    stop_word_cond = False
                    for stop_word in stop_words:
                        stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):]==stop_word)

                output_ids = generation_output.sequences[0]


                raw_output = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)
                result_output = process_text_output(raw_output)

                try:
                    code_output = round(float(eval(code_output))) % 1000
                except Exception as e:
                    yield f"{e}\nfinal_eval\n"
                    code_output = -1

            except Exception as e:
                yield f"{e}\nError in solving\n"
                result_output, code_output = -1, -1

            if code_output!=-1:
                outputs.append(code_output)
                code_answers+=1

            if result_output!=-1:
                outputs.append(result_output)
                text_answers+=1

            if len(outputs) > 0:
                occurances = Counter(outputs).most_common()
                yield f"{occurances}\n"
                if occurances[0][1] > best_count:
                    yield "GOOD ANSWER UPDATED!\n"
                    best = occurances[0][0]
                    best_count = occurances[0][1]
                    best_jj = jj
                if occurances[0][1] > 5:
                    yield "ANSWER FOUND!\n"
                    break

            results.append(result_output)
            answers.append(code_output)
            all_outputs.append(decoded_output)

            best_stats[i] = (best, best_count) 
            question_type_counts[i] = (text_answers, code_answers)
            total_outputs[i] = outputs

            total_results[i] = results
            total_answers[i] = answers
            

            yield f"code_answers: {code_answers-starting_counts[1]} text_answers: {text_answers-starting_counts[0]}\n"
        print("All Outputs",all_outputs)
        print("Best Solution Sample:",best_solution)
        print("Sample best stat print:",best_stats)
        for k in range(answers):
            if answers[k] == best_stats[i][0]:
                best_solution = all_outputs[k]

        yield f"Predicted best answer: {best_stats}\n"
        
        
        
        with open('code.py', 'w') as fout:
            fout.write("print('done')")

        batcmd = 'timeout 7 ' + sys.executable + ' code.py'
        try:
            shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
            print(shell_output)
        except:
            pass

    except Exception as e:
        yield str(e)

def main():
    # Initialize session state for showing steps
    if "show_steps" not in st.session_state:
        st.session_state.show_steps = False

    # Page layout
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container {
            max-width: 1600px;  /* Increase the max-width to make the page wider */
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        .box {
            padding: 1rem;
            border: 1px solid #4B0082;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .scroll-box {
            max-height: 500px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>Olympiads Mathematics Question Solver</h1>", unsafe_allow_html=True)

    # Input area
    st.markdown("<h2 style='color: #4B0082;'>Input Your Question Below</h2>", unsafe_allow_html=True)
    question = st.text_area("", placeholder="e.g., x^2 - 4 = 0")

    if st.button("Solve"):
        if question:
            # Progress bar
            progress_bar = st.progress(0)

            # Divide the screen into two columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h2 style='color: #4B0082;'>Steps</h2>", unsafe_allow_html=True)
                steps_area = st.empty()

            with col2:
                st.markdown("<h2 style='color: #4B0082;'>Final Answer</h2>", unsafe_allow_html=True)
                final_solution_area = st.empty()
                best_stat_area = st.empty()

            with st.spinner("Solving..."):
                progress = 0
                all_steps = []

                n_repetitions = 15  # Adjust based on actual solving logic
                for jj, intermediate_result in enumerate(solve_math_question(question)):
                    all_steps.append(intermediate_result)
                    with col1:
                        steps_area.markdown('<div class="box scroll-box">' + '<br>'.join(all_steps) + '</div>', unsafe_allow_html=True)
                    progress = min((jj + 1) / n_repetitions, 1.0)  # Clamp progress to a maximum of 1.0
                    progress_bar.progress(progress)

                # Ensure to capture the final output after the loop
                with col1:
                    st.markdown('<div class="box">Solving completed.</div>', unsafe_allow_html=True)
                
                # Display the best stat and final solution separately in the right column
                best_stat = best_stats.get(0, "No best stat found")
                with col2:
                    final_solution_area.markdown(
                        '<div class="box"><h3 style="color: #4B0082;">Best Solution:</h3>' + str(best_solution) + '</div>',
                        unsafe_allow_html=True)
                    best_stat_area.markdown(
                        '<div class="box"><h3 style="color: #4B0082;">Answer:</h3>' + str(best_stat[0]) + '</div>',
                        unsafe_allow_html=True)

        else:
            st.error("Please write a question before solving.")

    # Footer
    #st.markdown("<h3 style='text-align: center; color: #4B0082;'>Powered by DeepSeek & Streamlit</h3>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
