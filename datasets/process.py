import pandas as pd
import json
import random
import re

if __name__ == '__main__':
    # # Path to your Parquet file
    # file_path = 'datasets/ASDiv.parquet'

    # # Read the Parquet file into a pandas DataFrame
    # df = pd.read_parquet(file_path)

    # # Display the first few rows of the DataFrame
    # # print(df.head())

    # # Convert the DataFrame to a JSON string
    # # orient='records' is often the most useful format, with each row as a dictionary
    # json_output = df.to_json(orient='records')

    # # Print the JSON output to the console
    # with open('datasets/asdiv_test.json', 'w') as f:
    #     df.to_json(f, orient='records', indent=4) # indent=4 makes the file human-readable

    input_file = "aime2026.jsonl"
    output_file = "aime2026_test.json"

    data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            item = json.loads(line)
            
            new_item = {
                "index": idx,
                "question": item["problem"],  # rename field
                "answer": item["answer"]
            }
            
            data.append(new_item)

    # Save as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

  




    # new_data = []
    # with open('datasets/SVAMP.json','r') as f:
    #     data = json.load(f)
     

    # for i,item in enumerate(data):
    #     new_single_data = {}
    #     new_single_data['index'] = i
    #     new_single_data['question'] = item['Body'] + ' ' + item['Question']
    #     new_single_data['answer'] = item['Answer']


    #     new_data.append(new_single_data)

    # with open('datasets/svamp_test.json','w') as f:
    #     json.dump(new_data,f,indent=4)
            





    # with open('datasets/strategy.jsonl','r') as f:
    #     data = f.readlines()

    # data = [json.loads(i) for i in data]
    # for i in range(len(data)):
    #     data[i].update({'index':i})



    # with open('datasets/strategy.json','w') as f:
    #     json.dump(data,f,indent=4)





    # random.seed(1)
    # new_data = []
    # with open('datasets/MathQA.json','r') as f:
    #     data = json.load(f)
     
    
    # for i,item in enumerate(data):
    #     new_single_data = {}
    #     new_single_data['index'] = i
    #     new_single_data['question'] = item['Problem']
    #     correct_option = item['correct']
    #     options = item['options'].split(',')

    #     for option in options:
    #         if correct_option in option:
    #             new_single_data['answer'] = option.split(f'{correct_option} )')[-1].strip()
 
    #     if re.match(r'[-+]?\d+(\.\d+)?$',new_single_data['answer']):
    #         new_data.append(new_single_data)


    # random.shuffle(new_data)
    # new_data = new_data[:500]
    # for i,item in enumerate(new_data):
    #     item['index'] = i

    # with open('datasets/mathqa_test.json','w') as f:
    #     json.dump(new_data,f,indent=4)