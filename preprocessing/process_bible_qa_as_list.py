import csv
import re
from collections import OrderedDict
import json

"""
global variables
"""
abbreviation = OrderedDict()
kjv = OrderedDict()
asv = OrderedDict()
ylt = OrderedDict()
web = OrderedDict()


def read_bible():
    abbre_dir = "../data/bible/key_abbreviations_english.csv"
    kjv_dir = "../data/bible/t_kjv.csv"
    asv_dir = "../data/bible/t_asv.csv"
    ylt_dir = "../data/bible/t_ylt.csv"
    web_dir = "../data/bible/t_web.csv"

    #build abbreviation dictionary
    with open(abbre_dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for id, a, b, p in reader:
            abbreviation[a] = str(b)

    #build kjv

    with open(kjv_dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for id, book, chap, verse, text in reader:
            #pad to 8 digits
            kjv[str(id).rjust(8, '0')] = text

    #build asv
    with open(asv_dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for id, book, chap, verse, text in reader:
            #pad to 8 digits
            asv[str(id).rjust(8, '0')] = text

    #build ylt
    with open(ylt_dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for id, book, chap, verse, text in reader:
            #pad to 8 digits
            ylt[str(id).rjust(8, '0')] = text

    # build ylt
    with open(web_dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for id, book, chap, verse, text in reader:
            # pad to 8 digits
            web[str(id).rjust(8, '0')] = text



def process_q(input):
    dot_index = input.find(".")
    question = input[dot_index+1:].strip()
    #print("question", question)
    return question
    #finish this


def find_verse_code(whole_verse):

    space_index = whole_verse.rfind(" ")
    colon_index = whole_verse.find(":")

    book = whole_verse[: space_index].strip()
    chapter = whole_verse[space_index + 1 : colon_index]
    verse = whole_verse[colon_index + 1 :]

    #print("book name: ", book)
    #print ("dic book: ", abbreviation[book])
    book_code = abbreviation[book].rjust(2, '0')


    chapter_code = chapter.rjust(3, '0')
    verse_code = verse.rjust(3, '0')

    entire_verse_code = book_code + chapter_code + verse_code

    return(entire_verse_code)


def clean_sentence(text):
    #print("before: ", text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(' +', ' ', text)

    #print("after: ", text)

    return text

def get_verses_and_labels(verse_code):

    #take the two verses before as negative examples

    kjv_sentences = []
    asv_sentences = []
    ylt_sentences = []
    web_sentences = []
    labels = []

    book = verse_code[0:2]
    chapter = verse_code[2:5]

    #print(verse_code)
    book_beginning = int(book + chapter + "001")
    next_chapter = int(book + str(int(chapter) + 1).rjust(3, '0') + "000")

    ''' this is for getting 3 '''
    lower_bound = int(verse_code) - 2
    upper_bound = int(verse_code) + 1

    if int(verse_code) < book_beginning + 2:
        #if the verse is the first or second of the chapter
        lower_bound = int(verse_code)
        upper_bound = int(verse_code)+3
        


    '''for getting 10
    
    lower_bound = int(verse_code) - 9
    upper_bound = int(verse_code) + 1

    if lower_bound < book_beginning:
        #lower bound is smaller than book beginin
        lower_bound = book_beginning
        upper_bound = book_beginning + 10

    '''

    for current_verse in range(lower_bound, upper_bound):
        #check if verse code is valid in all books
        key = str(current_verse).rjust(8, '0')
        if key in kjv and key in asv and key in ylt and key in web:
            #kjv_sentences = kjv_sentences + [clean_sentence(kjv[key])]
            #asv_sentences = asv_sentences + [clean_sentence(asv[key])]
            #ylt_sentences = ylt_sentences + [clean_sentence(ylt[key])]
            web_sentences = web_sentences + [clean_sentence(web[key])]

            label = 0
            if int(current_verse) == int(verse_code):
                label = 1

            labels = labels + [label]
        else:
            break

    #return [[kjv_sentences, asv_sentences, ylt_sentences, web_sentences],labels]
    return [[web_sentences], labels]

def process_a(input):
    result = []
    sentences = []
    labels = []

    open_bracket_index = input.rfind("(")
    close_bracket_index = input.rfind(")")
    dot_index = input.find(".")

    #literal_answer = input[:open_bracket_index][dot_index+1:].strip()

    entire_verse = input[open_bracket_index+1:close_bracket_index].strip()

    #answer = entire_verse[0]
    #print("literal answer", literal_answer)
    #print("entire verse", entire_verse)

    verse_code = find_verse_code(entire_verse)

    #kjv_verse = clean_sentence(kjv[verse_code])
    #asv_verse = clean_sentence(asv[verse_code])
    #ylt_verse = clean_sentence(ylt[verse_code])
    #web_verse = clean_sentence(web[verse_code])
    #label = 1

    #appending the correct verse
    #result.append([verse_code, kjv_verse, asv_verse, ylt_verse, web_verse, label])

    verses_and_labels = get_verses_and_labels(verse_code)
    sentences = verses_and_labels[0]
    labels = verses_and_labels[1]

    #add all irrelevant verses
    #for verse in irrelevant_verses_in_four_versions:
    #    result.append(verse)
        #print(verse)

    #print(result)
    #a list of lists for sentences, and a list for labels
    return [sentences, labels]


def read_data():
    dir = "../data/bible_qa/1001.csv"

    list_bible_qa = []

    with open(dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter = "\t")
        next(reader)
        id = 1
        for question, answer in reader:
            qa_pair = {}
            processed_q = process_q(question)
            processed_a = process_a(answer)
            answers = processed_a[0]
            labels = processed_a[1]

            answer_num = len(labels)
            question_list = [processed_q] * answer_num

            qa_pair["question"] = question_list
            qa_pair["answers"] = answers #has a list of answers, each one is a different translation
            qa_pair["labels"] = labels
            #if (processed_a == False):
            #    continue
            list_bible_qa.append(qa_pair)

    print("finished processing")
    #print(bible_qa[0])
    #print(len(bible_qa))
    #print(bible_qa[0][0])

    with open('../data/bible_qa/bible_qa_list_3_ylt.json', 'w') as f:
        json.dump(list_bible_qa, f)

if __name__ == '__main__':

    read_bible()
    read_data()