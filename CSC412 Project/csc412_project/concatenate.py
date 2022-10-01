from bpe import *
from dvae_sample import *
import os

if __name__ == '__main__':
    token_limit = 256

    # dVae
    print("Initializing dVae")
    dev = torch.device('cpu')
    enc = load_model("https://cdn.openai.com/dall-e/encoder.pkl", dev)
    print("dVae ready")

    # BPE
    print("Token limit is", token_limit, "tokens")
    print("Initializing BPE")
    caption = ['./caption.txt']

    word_dict = get_word(caption)
    tokens_freq, tokens_dict = word_to_token(word_dict)
    while len(tokens_freq.keys()) < token_limit:
        pairs = get_stats(word_dict)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        word_dict = merge_word(best, word_dict)
        tokens_freq, tokens_dict = word_to_token(word_dict)
    sorted_tokens_tuple = sorted(tokens_freq.items(), key=lambda item: (token_len(item[0]), item[1]), reverse=True)
    sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
    print("BPE complete")
    print("The result tokens are ", sorted_tokens)

    print("Tokenize captions and concatenate with image")
    image_to_caption_file = open('image_to_caption.txt', 'r')
    image_to_caption_lines = image_to_caption_file.readlines()
    for line in image_to_caption_lines:
        image_index = line[2:7]
        dir = './image to vector/' + image_index + '.txt'
        if os.path.exists(dir):
            continue

        image_captions = line[11:-2]
        image_to_caption_string = image_captions.replace('\n', ' ')
        image_to_token = tokenize_word(image_to_caption_string, sorted_tokens=sorted_tokens, unknown_token='</u>')
        image_to_token_vector = token_to_vector(sorted_tokens, image_to_token)

        x = preprocess(open_image('./image/image_' + image_index + '.jpg'))
        z_logits = enc(x)
        z = ((torch.argmax(z_logits, axis=1)).numpy())[0]
        image_vector = np.reshape(z, 1024)

        concatenated_vector = np.concatenate((image_to_token_vector, image_vector), axis=None)
        image_to_vector = open(dir, 'w')
        np.savetxt(image_to_vector, concatenated_vector, fmt='%d')
        image_to_vector.close()
        print(image_index)

    image_to_caption_file.close()

    # file = open('./image_to_vector.txt', 'r')
    # lines = file.readlines()
    # for line in lines:
    #     print(line)

