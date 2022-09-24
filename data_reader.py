# Python 3 or above

import re
import numpy as np


def read_review_data(filelist):
    '''
    This function is to extract reviews from datasets
    '''
    for file in filelist:
        # Open file and read lines
        with open(file, encoding="ISO-8859-1") as f:
            lines = f.read().splitlines()

        datalist = []

        for line in lines:
            # Remove length, rating
            line = line.split(" ", 2)[2]
            # Add split characters to regex based on requirement
            line = re.sub(r'[\.\?]', ",", line)
            reviews = line.split(",")

            for review in reviews:

                # If review is empty ignore
                if review == "":
                    continue

                # If review contains non-alphabets ignore
                if re.match(r'[^a-zA-Z]', review) is not None:
                    continue

                datalist.append(review)

    # Convert to numpy array
    ret_val = np.array(datalist)
    return ret_val


def read_combine_data(filelist):
    '''
    This function is to read data including ratings
    '''
    for file in filelist:
        # Open file and read lines
        with open(file, encoding="ISO-8859-1") as f:
            lines = f.read().splitlines()

        datalist = []

        for line in lines:
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

                # If review is empty ignore
                if review == "":
                    continue

                # If review contains non-alphabets ignore
                if re.match(r'[^a-zA-Z]', review) is not None:
                    continue

                datalist.append(str(rating)+' [SEP] '+review)

    # Convert to numpy array
    ret_val = np.array(datalist)
    return ret_val