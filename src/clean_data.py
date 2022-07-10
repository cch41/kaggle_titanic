import numpy as np


def embarked(s):
    if "Q" in s:
        return 1
    elif "C" in s:
        return 2
    else:
        return 3


# TRAINING DATA
imported_training_data = np.genfromtxt('data/train.csv',
                                       delimiter=',',
                                       autostrip=True,
                                       skip_header=1,
                                       usecols=(2, 5, 6, 7, 8, 10),
                                       encoding='UTF-8',
                                       converters={
                                           2: lambda s: int(s or 0),
                                           5: lambda s: 1 if "female" in s else 0,
                                           6: lambda s: round(float(s or 0), 1),
                                           7: lambda s: int(s or 0),
                                           8: lambda s: int(s or 0),
                                           10: lambda s: round(float(s or 0), 2),
                                           12: embarked,
                                       })

array_training_data = []
for i, row in enumerate(imported_training_data):
    array_training_data.append([*row])

# FOR EXPORT
training_solution = np.genfromtxt('data/train.csv',
                                  delimiter=',',
                                  skip_header=1,
                                  usecols=(1),
                                  converters={1: lambda s: int(s or 0)}
                                  )
training_data = list(zip(np.array(array_training_data), training_solution))
# we are exporting an array of length two, with each entry being a several hundred set of data
# instead, we should zip these together so we export a several hundred line array with zipped lines

# TEST DATA
imported_test_data = np.genfromtxt('data/train.csv',
                                   delimiter=',',
                                   autostrip=True,
                                   skip_header=1,
                                   usecols=(2, 5, 6, 7, 8, 10, 12),
                                   encoding='UTF-8',
                                   converters={
                                       2: lambda s: int(s or 0),
                                       5: lambda s: 1 if "female" in s else 0,
                                       6: lambda s: round(float(s or 0), 1),
                                       7: lambda s: int(s or 0),
                                       8: lambda s: int(s or 0),
                                       10: lambda s: round(float(s or 0), 2),
                                       12: embarked,
                                   })

array_test_data = []
for i, row in enumerate(imported_test_data):
    array_test_data.append([*row])

# FOR EXPORT
test_solution = np.genfromtxt('data/test.csv',
                              delimiter=',',
                              skip_header=1,
                              usecols=(1),
                              converters={1: lambda s: int(s or 0)}
                              )
test_data = list(zip(np.array(array_test_data), test_solution))
