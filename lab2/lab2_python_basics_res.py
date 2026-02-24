import random

n = 20;
a, b = 0, 100;
lst = [random.randint(a, b) for _ in range(n)]
print("spisok: \n", lst, "\n")


sum = 0;
for i in lst:
    if (i % 2 == 0):
        sum += i;

print("summa chetnih znacheniy: ", sum)