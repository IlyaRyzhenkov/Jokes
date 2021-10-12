import sys

import gensim.downloader
import stanza
import nltk
import requests

from nltk.corpus import wordnet as wn


STRINGS_TO_HANDLE = [
    "Losing Faith in the State, Some Mexican Towns Quietly Break Away",
    "Right and Left React to Questions About Trump’s Mental State",
    "Steve Bannon Steps Down From Breitbart Post",
    "Trump Administration Says States May Impose Work Requirements for Medicaid",
    "Can Requiring People to Work Make Them Healthier?",
    "She Left France to Fight in Syria. Now She Wants to Return. But Can She?",
    "Facebook Overhauls News Feed to Focus on What Friends and Family Share",
    "Wildlife Detectives Pursue the Case of Dwindling Elephants in Indonesia",
    "Hawaii Panics After Alert About Incoming Missile Is Sent in Error",
    "Military Quietly Prepares for a Last Resort: War With North Korea",
    "One Year After Women’s March, More Activism but Less Unity",
    "Kazakhstan Cheers New Alphabet, Except for All Those Apostrophes",
    "Pentagon Suggests Countering Devastating Cyberattacks With Nuclear Arms",
    "Horror for 13 California Siblings Hidden by Veneer of a Private Home School",
    "North and South Korean Teams to March as One at Olympics",
    "Apple, Capitalizing on New Tax Law, Plans to Bring Billions in Cash Back to U.S.",
    "Senate Democrats Make Hard Turn Left in Warming Up for 2020 Race",
    "Tax Overhaul Is a Blow to Affordable Housing Efforts",
    "How Trump and Schumer Came Close to a Deal Over Cheeseburgers",
    "Statue of Liberty Will Reopen Despite Government Shutdown",
    "Venezuela’s Most-Wanted Rebel Shared His Story, Just Before Death"
]


def find_word_to_replace(wv, words):
    for word in reversed(words):
        if word.pos in ("NOUN", "ADJ"):
            try:
                r = wv.similarity(word.lemma, "cat")
                return word.text, word.lemma, word.pos
            except KeyError as e:
                continue
    raise ValueError("Can't find a word to replace")


def find_antonyms(word, type):
    if type == "ADJ":
        type = wn.ADJ
    elif type == "NOUN":
        type = wn.NOUN
    else:
        type = None
    antonyms = []
    for i in wn.synsets(word, pos=type):
        for l in i.lemmas():
            for a in l.antonyms():
                antonyms.append(a.name())
    return antonyms


def get_rythms_and_similar(word):
    rythms = set()
    BASE_RYTHMS_URL = "https://api.datamuse.com/words?rel_rhy=" + word
    BASE_SIMILAR_URL = "https://api.datamuse.com/words?sl=" + word
    res = requests.get(BASE_RYTHMS_URL).json()
    for i in range(5):
        try:
            rythms.add(res[i]["word"])
        except IndexError as e:
            break

    res2 = requests.get(BASE_SIMILAR_URL).json()
    for i in range(5):
        try:
            rythms.add(res2[i]["word"])
        except IndexError as e:
            break
    return rythms


def find_unsimilar(wv, word, unsimilar_list):
    min_coef = 1000
    min_word = ''
    for uw in unsimilar_list:
        try:
            r = wv.similarity(word, uw)
        except KeyError as e:
            continue
        if r < min_coef:
            min_coef = r
            min_word = uw
    return min_word


def handle(wv, input_string):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', tokenize_no_ssplit=True)
    doc = nlp(input_string)
    words = []
    for sentence in doc.sentences:
        for word in sentence.words:
            words.append(word)
    word_text, word_to_replace, type = find_word_to_replace(wv, words)
    antonyms = find_antonyms(word_to_replace, type)
    rythms = get_rythms_and_similar(word_to_replace)
    antonyms.extend(rythms)
    res_word = find_unsimilar(wv, word_to_replace, antonyms)
    rep_str = res_word + '(' + word_text + ')'
    ans = rep_str.join(input_string.rsplit(word_text, 1))
    return ans


if __name__ == "__main__":
    if len(sys.argv) == 2:
        input_strings = (sys.argv[1],)
    else:
        input_strings = STRINGS_TO_HANDLE
    stanza.download("en")
    nltk.download('wordnet')
    word_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
    i = 1
    with open("examples.txt", "w") as f:
        for string in input_strings:
            try:
                res = handle(word_vectors, string)
                f.write(f"{i}) {res}\n")
                print(f"{i}) {res}\n")
                i += 1
            except ValueError as e:
                f.write(f"{string} - error\n")
                print(f"{string} - error\n")
                continue
