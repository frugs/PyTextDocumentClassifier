import string
import math
import functools
import numpy

punctuation_to_strip = string.punctuation.replace("#", "").replace("+", "").replace("-", "")


def normalise_word(word):
    return word.lstrip(punctuation_to_strip).rstrip(punctuation_to_strip).lower()


def word_list_from_document_content(document_content):
    return [normalise_word(word) for word in document_content.split()]


def words_from_file(file_object):
    for line in file_object:
        for word in word_list_from_document_content(line):
            yield word


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
