import numpy as np

my_data = np.genfromtxt('../data/train.csv',
                        delimiter=',',
                        autostrip=True,
                        skip_header=1,
                        usecols=(2, 5, 6, 7, 8, 10),
                        encoding='UTF-8',
                        converters={
                            2: lambda s: int(s or 0),
                            5: lambda s: 1 if "female" in s else 0,
                            6: lambda s: int(s or 0),
                            7: lambda s: int(s or 0),
                            8: lambda s: int(s or 0),
                            10: lambda s: int(s or 0)
                        })
for i in range(20):
    print(my_data[i])
