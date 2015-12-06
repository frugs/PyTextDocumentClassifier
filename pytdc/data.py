import string
import math
import functools
import numpy
import bs4

punctuation_to_strip = string.punctuation.replace("#", "").replace("+", "").replace("-", "")


def normalise_word(word):
    return word.lstrip(punctuation_to_strip).rstrip(punctuation_to_strip).lower()


def word_list_from_document_content(document_content):
    return [normalise_word(word) for word in document_content.split()]


def words_from_file(file_object):
    for line in file_object:
        for word in word_list_from_document_content(line):
            yield word


def words_from_email(message):
    def pair_part_content_type_with_part_content(part):
        return part.get_content_type(), words_from_email(part)

    if message.is_multipart():
        part_contents = [pair_part_content_type_with_part_content(part) for part in message.get_payload()]
        text_plain_parts = [part_content[1]
                            for part_content
                            in part_contents
                            if part_content[0].startswith("text/plain")]

        if len(text_plain_parts) > 0:
            return text_plain_parts[0]

        text_html_parts = [part_content[1]
                            for part_content
                            in part_contents
                            if part_content[0].startswith("text/html")]

        if len(text_html_parts) > 0:
            return text_html_parts[0]

        return []

    else:
        if "text/plain" in message.get_content_type() or "text/html" in message.get_content_type():
            payload = message.get_payload(decode=True)

            html = bs4.BeautifulSoup(payload)
            for style_elem in html.find_all("style"):
                style_elem.extract()

            lines = [line.strip() for line in html.text.splitlines()]
            line_words_in_lines = [line.split() for line in lines]
            return [normalise_word(word) for line_words in line_words_in_lines for word in line_words]


def create_classified_data_set(document_vectors, classification_vector):
    return ((document_vector, classification_vector) for document_vector in document_vectors)


def save_classified_data_set(data_set, filename):
    arrays = functools.reduce(lambda acc, data_entry: acc + list(data_entry), data_set, [])
    numpy.savez_compressed(filename, *arrays)


def load_classified_data_set(filename):
    with numpy.load(filename) as data:
        return [(data["arr_%d" % i], data["arr_%d" % (i + 1)]) for i in range(0, len(data.items()), 2)]


def vectorise_words_using_word_vector_model(words, word_vector_model, word_vector_dims):
    def vectorise_word(word):
        return word_vector_model[word] if word in word_vector_model.vocab.keys() else numpy.zeros(word_vector_dims)

    sum_vector = functools.reduce(lambda a, b: a + b, (vectorise_word(word) for word in words), numpy.zeros(word_vector_dims))
    vector_length = sum_vector.dot(sum_vector)
    return sum_vector / math.sqrt(vector_length) if vector_length > 0 else sum_vector
