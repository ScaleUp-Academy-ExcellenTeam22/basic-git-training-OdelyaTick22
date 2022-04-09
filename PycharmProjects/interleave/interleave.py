
def interleave_generator(*args: iter):
    """
    A function that receives one or more parameters
    that can be repeated, and returns a list of intertwined organs.
    :param args: one or more iterable parameters
    :return: List of intertwined organs
    """
    new_list = []
    while True:
        zip_item = zip(*args)
        for item in zip_item:
            new_list.extend(item)
        yield new_list


if __name__ == '__main__':
    gen = interleave_generator('abc', [1, 2, 3], ('!', '@', '#'))
    print(next(gen))
