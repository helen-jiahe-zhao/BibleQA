import csv
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

def process_a(input):

    open_bracket_index = input.rfind("(")
    close_bracket_index = input.rfind(")")
    dot_index = input.find(".")

    literal_answer = input[:open_bracket_index][dot_index+1:].strip()

    entire_verse = input[open_bracket_index+1:close_bracket_index].strip()

    #answer = entire_verse[0]
    #print("literal answer", literal_answer)
    #print("entire verse", entire_verse)

    verse_code = find_verse_code(entire_verse)

    all_context = get_context_from_verse(verse_code)
    kjv_context = all_context[0]
    asv_context = all_context[1]
    ylt_context = all_context[2]
    web_context = all_context[3]

    kjv_verse = kjv[verse_code]
    asv_verse = asv[verse_code]
    ylt_verse = ylt[verse_code]
    web_verse = web[verse_code]

    return [kjv_context, asv_context, ylt_context, web_context, literal_answer, verse_code, kjv_verse, asv_verse, ylt_verse, web_verse]


def read_data():
    dir = "data/bible_qa/1001.csv"

    #format:
    #question, context KJV, context ASV, context YLT, context WEB
    #literal answer, verse number,
    ##verse content in 4 translations
    #7 columns for each question

    bible_qa = []

    with open(dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter = ",")
        next(reader)
        id = 1
        for question, answer in reader:
            processed_q = process_q(question)
            processed_a = process_a(answer)
            if (processed_a == False):
                continue
            processed_qa = [str(id)] + processed_q + processed_a
            id = id + 1
            #print(processed_qa)
            bible_qa.append(processed_qa)

    print("finished processing")
    #print(bible_qa[0])
    #print(len(bible_qa))
    #print(bible_qa[0][0])

    with open('bible_qa.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['ID', 'Question', 'KJV_Context', 'ASV_Context', 'YLT_Context', 'WEB_Context', 'Answer', 'Verse_Code', 'KJV_Verse', 'ASV_Verse', 'YLT_Verse', 'WEB_Context'])
        for row in bible_qa:
            writer.writerow(row)
        print("done")

if __name__ == '__main__':

    read_bible()
    read_data()