import os

if __name__ == '__main__':
    caption = open('caption.txt', 'w')
    image_to_caption = open('image_to_caption.txt', 'w')

    d = './caption/text_c10/'
    for file in os.listdir(d):
        if 't7' not in file and 'class' in file:
            class_file = os.path.join(d, file) + '/'
            for image_file in os.listdir(class_file):
                if 'txt' in image_file:
                    complete_path = os.path.join(class_file, image_file)
                    image_index = image_file[-9:-4]
                    image_txt = open(complete_path, 'r')
                    content = image_txt.readlines()
                    caption_string = ''
                    for line in content:
                        if '{' in line or '[' in line or ']' in line:
                            continue
                        caption_string += line
                        caption.write(line)
                    image_to_caption.write(str((image_index, caption_string)) + '\n')
                    image_txt.close()
    caption.close()
    image_to_caption.close()
