"""Just for fun"""
import functools
import itertools
from typing import Any, Iterable, Callable, Generator, Generic, Optional, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class Stream(Generic[T]):
    source: Iterable[T]

    def __init__(self, source: Iterable[T]):
        self.source = source

    def __iter__(self):
        return self.source.__iter__()

    def __next__(self):
        for x in self.source:
            yield x

    @classmethod
    def of(cls, *items):
        return cls(items)

    def map(self, function: Callable[[T], U]) -> "Stream[U]":
        return self.__class__(function(x) for x in self.source)

    def reduce(self, function: Callable[[T, U], T], initial: Optional[U] = None) -> T:
        return functools.reduce(function, self, initial)

    def filter(self, function: Callable[[T], T]) -> "Stream[U]":
        return self.__class__(x for x in self.source if function(x))

    def distinct(self):
        seen = set()
        def unseen(x):
            if x in seen:
                return False
            seen.add(x)
            return True
        return Stream(x for x in self.source if unseen(x))

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None):
        return self.__class__(itertools.islice(self, start, stop, step))

    def chain(self):
        return self.__class__(itertools.chain.from_iterable(self))

    def flat_map(self, function: Callable[[T], Iterable[U]]) -> "Stream[U]":
        return self.map(function).chain()

    def not_none(self):
        return self.filter(lambda x: x is not None)

    def collect(self, function: Callable[[Iterable[T]], U]) -> U:
        return function(self.source.__iter__())


class StreamSkipNones(Stream):
    def __init__(self, source: Iterable[T] | Generator["Stream[T]", Any, None]):
        super().__init__(x for x in source if x is not None)


def main():
    for x in Stream([1, 2, 3, None]).not_none().map(lambda x: x * 2).map(lambda x: x + 4):
        print(x)
    for x in StreamSkipNones([4, 5, 6, None]).map(lambda x: x * 7).map(lambda x: x + 4):
        print(x)
    print(list(Stream.of(1,2,3).map(lambda x: x * 2)))
    print(list(Stream.of("a", "b")))
    print(Stream.of("a", "b").collect(list))
    stream = Stream.of("c", "d").map(lambda x: f"{x}y{x}")
    print("a".join(stream))
    print("z".join(stream))
    for x in Stream([[666, 2, 3], [4, 5, 6]]).flat_map(lambda x: x):
        print(x)

    for x in Stream.of("a", "b", "c").flat_map(lambda x: [x] * 10):
        print(x)


if __name__ == "__main__":
    main()
