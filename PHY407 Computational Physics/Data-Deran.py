import random
import pylab as plt
import numpy as np

test = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13',
        'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13',
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
        'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13']
random.shuffle(test)

# plan 1
# def check(deck):
#     temp = []
#     count = 0
#     for i in range(len(deck)):
#         if len(temp) == 8:
#             return count
#         else:
#             inside = False
#             for item in temp:
#                 if item[1:] == deck[i][1:]:
#                     temp.remove(item)
#                     inside = True
#             if not inside:
#                 temp.append(deck[i])
#             count += 1
#     return count


# plan 2
N = 10000


def check(deck):
    random.shuffle(deck)
    temp = []
    for i in range(len(deck)):
        for item in temp:
            if item[1:] == deck[i][1:]:
                return len(temp)
        temp.append(deck[i])


arr_test = np.zeros(N)
for i in range(N):
    arr_test[i] = check(test)

plt.hist(arr_test, bins=30, range=(1, 13), label='Histogram', density=True, stacked=True)
plt.show()

