from collections import Counter

a = str([4 for _ in range(1000)])
b = tuple([4 for _ in range(1000)])
import sys

print("Size of list a:", sys.getsizeof(a))
print("Size of Counter b:", sys.getsizeof(b))
