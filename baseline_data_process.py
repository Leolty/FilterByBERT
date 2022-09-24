# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:32:15 2022

@author: Tianyang Liu
"""

# Use python 3 or above

import re               # Regular Expressions
import numpy as np      # Numpy
from sklearn.naive_bayes import BernoulliNB

# Extracts words from given list of files and assigns an id to them
# Arguments: List of filenames
# Returns: Dictionary with key = word, value = id
def extract_words_and_add_to_dict(filenamelist):

    # Ignore these words
    stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being",
                  "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                  "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive",
                  "if", "in", "into", "is", "it", "its", "its", "itself", "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
                  "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should", "so",
                  "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre",
                  "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were", "what", "whats", "when",
                  "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with", "would", "you", "youd", "youll", "youre", "youve",
                  "your", "yours", "yourself", "yourselves", "need", "needed", "can", "u", "every", "rather", "gonna", "m", "tap", "fish", "gives", "nice", "hoping"]


    # Dictionary containing word to id mapping
    word_id = {}
    # For all files in the given list of filenames
    for filename in filenamelist:

        # Open file and read lines
        with open(filename, encoding="ISO-8859-1") as f:
            lines = f.read().splitlines()

        # For all lines in this file
        for line in lines:
            # Remove everything other than alphabets, whitespace
            line = re.sub(r'[^a-zA-Z\s]', "", line)
            # Split at whitespace
            words = line.split()
            # Ignore length, rating
            words = words[2:]
            for word in words:
                word = word.lower()
                # Add this word to dictionary if it is not a stop word
                if word not in stop_words:
                    word_id[word] = 0

        # Assign id's to words in dictionary
        id = 0
        for word in word_id:
            word_id[word] = id
            id += 1

    # Return the dictionary
    return word_id

# Reads data from given files and stores it in a numpy array
# Arguments: List of filenames, dictionary containing word -> id mapping
# Returns: 2D Numpy array representation of given data
# The first 5 columns represent the rating of the review
# Other columns represent which words are present in the review
def get_data(filenamelist, word_id):

    # Number of attributes per instance
    # These include bitstring for words, rating
    rating_bits = 6
    cols = rating_bits + len(word_id)

    # List representation of given data
    data_list = []

    # For all files in the given list of filenames
    for filename in filenamelist:
        # Open file and read lines
        with open(filename, encoding="ISO-8859-1") as f:
            lines = f.read().splitlines()

        for line in lines:
            line = line.lower()

            # Get rating
            rating = 0
            try:
                rating_str = line.split(" ", 2)[1]
            except IndexError:
                continue

            if rating_str.endswith("one"):
                rating = 1
            elif rating_str.endswith("two"):
                rating = 2
            elif rating_str.endswith("three"):
                rating = 3
            elif rating_str.endswith("four"):
                rating = 4
            elif rating_str.endswith("five"):
                rating = 5

            # Remove length, rating
            line = line.split(" ", 2)[2]
            # Add split characters to regex based on requirement
            line = re.sub(r'[\.\?]', ",", line)
            reviews = line.split(",")

            for review in reviews:

#                # If review is empty ignore
#                if review == "":
#                    continue
#
#                # If review contains non-alphabets ignore
#                if re.match(r'[^a-zA-Z]', review) is not None:
#                    continue

                # Create row
                instance = np.zeros((cols, ), dtype=int)
                # Set appropriate rating bit
                instance[rating] = 1

                words = review.split()
                for word in words:
                    attr_idx = word_id.get(word, None)
                    if attr_idx is not None:
                        instance[rating_bits + attr_idx] = 1

                # If instance is not all zeros then append
                # if np.count_nonzero(instance[rating_bits:]) != 0:
                data_list.append(instance)

    # Convert to numpy array
    ret_val = np.array(data_list)
    return ret_val

def get_reverse_mapping(mapping):
    return {val : key for key,val in mapping.items()}
