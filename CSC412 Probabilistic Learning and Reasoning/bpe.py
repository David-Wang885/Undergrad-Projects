import re
import collections
import numpy as np


def get_word(filename):
    words = collections.defaultdict()
    for text_file in filename:
        open_file = open(text_file, 'r', encoding='utf-8')
        file_lines = open_file.readlines()
        for line in file_lines:
            word = line.lower().strip().split()
            for item in word:
                word = ' '.join(list(item)) + ' </w>'
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
    return words


def get_stats(words):
    pair = collections.defaultdict()
    for word, freq in words.items():
        letters = word.lower().split()
        for i in range(len(letters) - 1):
            temp = (letters[i], letters[i + 1])
            if temp in pair.keys():
                pair[temp] += 1
            else:
                pair[temp] = 1
    return pair


def merge_word(pair, words):
    word_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in words:
        w = p.sub(''.join(pair), word)
        word_out[w] = words[word]
    return word_out


def get_tokens(words):
    tokens = collections.defaultdict(int)
    for word, freq in words.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens


def word_to_token(words):
    tokens_freq = collections.defaultdict(int)
    tokens_dict = {}
    for word, freq in words.items():
        word_token = word.split()
        for token in word_token:
            tokens_freq[token] += freq
        tokens_dict[''.join(word_token)] = word_token
    return tokens_freq, tokens_dict


def token_len(token):
    if '</w>' in token:
        return len(token[:-4]) + 1
    else:
        return len(token)


def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    if string == '':
        return []
    if not sorted_tokens:
        return [unknown_token]

    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace('.', '[.]'))

        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i + 1:],
                                           unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i + 1:],
                                       unknown_token=unknown_token)
        break
    return string_tokens


def token_to_vector(list_tokens, word_tokens):
    length = len(list_tokens)
    vector = np.zeros(length)
    for token in word_tokens:
        vector[list_tokens.index(token)] += 1
    return vector


if __name__ == '__main__':
    # file = ["./training-test/test1.txt", "./training-test/test2.txt", "./training-test/test3.txt",
    #         "./training-test/test4.txt", "./training-test/test5.txt", "./training-test/test6.txt",
    #         "./training-test/test7.txt", "./training-test/test8.txt", "./training-test/test9.txt",
    #         "./training-test/test10.txt"]
    file = ['./caption.txt']
    word_dict = get_word(file)

    print('==========')
    print('Tokens Before BPE')
    tokens_freq, tokens_dict = word_to_token(word_dict)
    print('All tokens: {}'.format(tokens_freq.keys()))
    print('Number of tokens: {}'.format(len(tokens_freq.keys())))
    print('==========')

    token_limit = 256
    while len(tokens_freq.keys()) < token_limit:
        pairs = get_stats(word_dict)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        word_dict = merge_word(best, word_dict)
        # print('Iter: {}'.format(i))
        # print('Best pair: {}'.format(best))
        tokens_freq, tokens_dict = word_to_token(word_dict)
        print('All tokens: {}'.format(tokens_freq.keys()))
        print('Number of tokens: {}'.format(len(tokens_freq.keys())))
        print('==========')

    sorted_tokens_tuple = sorted(tokens_freq.items(), key=lambda item: (token_len(item[0]), item[1]), reverse=True)
    sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]

    print(sorted_tokens)

    word_given = 'blueandredflowerwithgreenpetal</w>'
    print('Tokenizing word: {}...'.format(word_given))
    if word_given in tokens_dict:
        print('Tokenization of the known word:')
        print(tokens_dict[word_given])
        print('Tokenization treating the known word as unknown:')
        print(tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>'))
    else:
        print('Tokenizating of the unknown word:')
        result_tokens = tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>')
        print(result_tokens)
        print(token_to_vector(sorted_tokens, result_tokens))

