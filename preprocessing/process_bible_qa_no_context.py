import csv
import re
from collections import OrderedDict


"""
global variables
"""
abbreviation = OrderedDict()
kjv = OrderedDict()
asv = OrderedDict()
ylt = OrderedDict()
web = OrderedDict()


def read_bible():
    abbre_dir = "data/bible/key_abbreviations_english.csv"
    kjv_dir = "data/bible/t_kjv.csv"
    asv_dir = "data/bible/t_asv.csv"
    ylt_dir = "data/bible/t_ylt.csv"
    web_dir = "data/bible/t_web.csv"

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


                    #for key, value in bbe.items():
    #   print(key + " : " + value)

def process_q(input):
    dot_index = input.find(".")
    question = input[dot_index+1:].strip()
    #print("question", question)
    return [question]
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


def get_context_from_verse(verse_code):
    book = verse_code[0:2]
    chapter = verse_code[2:5]

    #print(verse_code)
    lower_bound = book + chapter + "000"
    upper_bound = book + str(int(chapter) + 1).rjust(3, '0') + "000"

    #print ("lower ", lower_bound)
    #print ("upper ", upper_bound)

    kjv_context = ""
    asv_context = ""
    ylt_context = ""
    web_context = ""

    for key, value in kjv.items():
        if int(key) > int(lower_bound) and int(key) < int(upper_bound):
            kjv_context = kjv_context + " " + value

    for key, value in asv.items():
        if int(key) > int(lower_bound) and int(key) < int(upper_bound):
            asv_context = asv_context + " " + value

    for key, value in ylt.items():
        if int(key) > int(lower_bound) and int(key) < int(upper_bound):
            ylt_context = ylt_context + " " + value

    for key, value in web.items():
        if int(key) > int(lower_bound) and int(key) < int(upper_bound):
            web_context = web_context + " " + value

    return [kjv_context.strip(), asv_context.strip(), ylt_context.strip(), web_context.strip()]

def clean_sentence(text):
    print("before: ", text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(' +', ' ', text)

    print("after: ", text)

    return text

def retrieve_irrelevant_verses(verse_code):
    #take the two verses before as negative examples

    result = []

    book = verse_code[0:2]
    chapter = verse_code[2:5]

    #print(verse_code)
    book_beginning = int(book + chapter + "001")
    #book_ending = int(book + str(int(chapter) + 1).rjust(3, '0') + "000")

    lower_bound = int(verse_code) - 2
    upper_bound = int(verse_code)

    if int(verse_code) < book_beginning + 2:
        #if the verse is the first or second of the chapter
        lower_bound = int(verse_code) + 1
        upper_bound = int(verse_code)+3

    #if int(verse_code) + 1 > book_ending:
    ##    upper_bound = book_ending
    #else:
    #    upper_bound = int(verse_code) + 1

    print("lower: ", lower_bound)
    print("upper: ", upper_bound)

    for current_verse in range(lower_bound, upper_bound):
        #check if verse code is valid in all books
        key = str(current_verse).rjust(8, '0')
        if key in kjv and key in asv and key in ylt and key in web:
            if int(current_verse) == int(verse_code):
                continue
            kjv_verse = clean_sentence(kjv[key])
            asv_verse = clean_sentence(asv[key])
            ylt_verse = clean_sentence(ylt[key])
            web_verse = clean_sentence(web[key])
            full_verse = [key, kjv_verse, asv_verse, ylt_verse, web_verse, 0]
            print("full verse:", full_verse)
            result.append(full_verse)
        else:
            break

    print("result", result)
    return result

def process_a(input):
    result = []
    open_bracket_index = input.rfind("(")
    close_bracket_index = input.rfind(")")
    dot_index = input.find(".")

    #literal_answer = input[:open_bracket_index][dot_index+1:].strip()

    entire_verse = input[open_bracket_index+1:close_bracket_index].strip()

    #answer = entire_verse[0]
    #print("literal answer", literal_answer)
    #print("entire verse", entire_verse)

    verse_code = find_verse_code(entire_verse)

    kjv_verse = clean_sentence(kjv[verse_code])
    asv_verse = clean_sentence(asv[verse_code])
    ylt_verse = clean_sentence(ylt[verse_code])
    web_verse = clean_sentence(web[verse_code])
    label = 1

    #appending the correct verse
    result.append([verse_code, kjv_verse, asv_verse, ylt_verse, web_verse, label])

    irrelevant_verses_in_four_versions = retrieve_irrelevant_verses(verse_code)

    #add all irrelevant verses
    for verse in irrelevant_verses_in_four_versions:
        result.append(verse)
        #print(verse)

    print(result)

    return result


def read_data():
    dir = "data/bible_qa/1001.csv"

    #format:
    #question, context KJV, context ASV, context YLT, context WEB
    #literal answer, verse number,
    ##verse content in 4 translations
    #7 columns for each question

    bible_qa = []

    with open(dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter = "\t")
        next(reader)
        id = 1
        question_num = 0
        for question, answer in reader:
            if question_num == 620:
                #roughly 70% of the dataset
                break
            question_num = question_num + 1
            processed_q = process_q(question)
            processed_a = process_a(answer)
            #if (processed_a == False):
            #    continue
            for processed_verse in processed_a:
                processed_qa = [str(id)] + processed_q + processed_verse
                id = id + 1
            #print(processed_qa)
                print("processed qa", processed_qa)
                bible_qa.append(processed_qa)

    print("finished processing")
    #print(bible_qa[0])
    #print(len(bible_qa))
    #print(bible_qa[0][0])

    with open('bible_qa_train.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['ID', 'Question', 'Verse_Code', 'KJV_Verse', 'ASV_Verse', 'YLT_Verse', 'WEB_Verse', 'Label'])
        for row in bible_qa:
            writer.writerow(row)
        print("done")

if __name__ == '__main__':

    read_bible()
    read_data()