from functools import wraps
from time import time


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        # print(f"{f.__name__} took {elapsed} seconds")
        return result, elapsed

    return wrapper


if __name__ == "__main__":

    @timed
    def test():
        print("test")
        return 1

    print(test())
