import os

if __name__ == '__main__':
    result = open('caption.txt', 'w')

    d = './caption/text_c10/'
    for file in os.listdir(d):
        if 't7' not in file and 'class' in file:
            class_file = os.path.join(d, file) + '/'
            for image_file in os.listdir(class_file):
                if 'txt' in image_file:
                    complete_path = os.path.join(class_file, image_file)
                    image_txt = open(complete_path, 'r')
                    content = image_txt.readlines()
                    for lines in content:
                        fixed_line = (lines.replace(',', '')).replace('.', '')
                        result.write(fixed_line)
                    image_txt.close()
    result.close()
