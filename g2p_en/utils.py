def chain_with_separator(iterables, sep):
    iterables = iter(iterables)
    yield from next(iterables)
    for x in iterables:
        yield sep
        yield from x
