import json


def get_json():
    with open('data/squad/dev-v1.1.json', 'r') as f:
        data = json.load(f)
        return data['data']

def process_text(data):
    #a list of context, question, answer, class:
    result = []

    for list in data:
        topic = list['paragraphs']
        for paragraph in topic:
            context = paragraph['context']
            qas = paragraph['qas']
            for pair in qas:
                question = pair['question']
                given_answer = pair['answers']

                result = get_answer_label_list(context, given_answer)
                answer_list = result[0]
                label_list = result[1]

                question_dict = {}
                question_dict['question'] = question
                question_dict['answers'] = answer_list
                question_dict['labels'] = label_list

                result.append(question_dict)

                """
                answer_sentences = get_answer_sentences(context, given_answer)

                #irrelevant_answer = get_irrelevant_answer(context, given_answer)
                result.append([context, question, answer_sentences[0], 1])
                if len(answer_sentences[1]) != 0:
                    #has irrelevant sentence
                    result.append([context, question, answer_sentences[1], 0])
                #else:
                    #print("hi")
                """

    json_result = {}
    json_result['data'] = result

    return json_result

def get_answer_label_list(context, given_answer):
    answer_list = context.split('.')
    label_list = [0] * len(answer_list)

    most_common_answer = ""
    # set the most common answer out of the 3
    if len(given_answer) > 2:
        if given_answer[1]['answer_start'] == given_answer[0]['answer_start']:
            most_common_answer = given_answer[0]
        else:
            most_common_answer = given_answer[2]
    else:
        most_common_answer = given_answer[0]

    # answers.append(answer['text'])
    all_indices = [i for i, ltr in enumerate(context) if ltr == '.']
    # print(context)
    answer_index = most_common_answer['answer_start']

    #found = False
    for num in range(0, len(all_indices)):
        if all_indices[num] > answer_index:
            label_list[num] = 1
            #found = True
            break

    #if not found:
    #    answer_list[-1] = 1

    return [answer_list, label_list]

def get_answer_sentences(context, given_answer):
    most_common_answer = ""
    #set the most common answer out of the 3
    if len(given_answer) > 2:
        if given_answer[1]['answer_start'] == given_answer[0]['answer_start']:
            most_common_answer = given_answer[0]
        else:
            most_common_answer = given_answer[2]
    else:
        most_common_answer = given_answer[0]

    #answers.append(answer['text'])
    all_indices = [i for i, ltr in enumerate(context) if ltr == '.']
    #print(context)
    answer_index = most_common_answer['answer_start']
    relevant_start_index = 0
    relevant_end_index = 0

    irrelevant_start_index = 0
    irrelevant_end_index = 0

    for num in range(0, len(all_indices)):
        if all_indices[num] > answer_index:
            if num == 0:
                relevant_start_index = 0
            else:
                relevant_start_index = all_indices[num - 1]

            relevant_end_index = all_indices[num]

            #use the sentence after as irrelevant sentence if the sentence is the first, otherwise use the sentence before
            if num == 0:
                irrelevant_start_index = all_indices[num]
                #if num == len(all_indices) - 1:
                    #last one already
                #    irrelevant_end_index = len(context)
                #else:
                if num + 1 == len(all_indices):
                    irrelevant_end_index = all_indices[num]
                else:
                    irrelevant_end_index = all_indices[num + 1]
            else:
                if num - 2 < 0:
                    #first one
                    irrelevant_start_index = 0
                else:
                    irrelevant_start_index = all_indices[num - 2]
                irrelevant_end_index = all_indices[num - 1]

    relevant_sentence = context[int(relevant_start_index) + 1:int(relevant_end_index)].strip()
    irrelevant_sentence = context[int(irrelevant_start_index) + 1:int(irrelevant_end_index)].strip()

    #context_in_sentences = context.split('.')
    #if len(all_indices) == 1:
    #    print("hi")
    return [relevant_sentence, irrelevant_sentence]

def write_json(data):
    with open('data/squad/dev-v1.1-sentence_list.json', 'w') as f:
        json.dump(data, f)


data = get_json()
json_result = process_text(data)
write_json(json_result)
