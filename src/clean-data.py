from numpy import genfromtxt, isnan
my_data = genfromtxt('../data/train.csv', delimiter=',', skip_header=1)

# for x in my_data:
#     count = 0
#     for y in x:
#         if isnan(y):
#             count+=1
#     print(count)