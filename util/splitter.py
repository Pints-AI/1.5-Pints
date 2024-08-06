class Splitter:

    """
    A class to manage splitting based on a specified ratio, indicating the frequency of False outcomes.
    A ratio of 0.9 will return 1 True for every 10 steps.
    """

    ratio: float
    numerator = 0
    denominator = 0

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def should_split(self):
        self.denominator += 1
        if self.numerator / self.denominator < self.ratio:
            self.numerator += 1
            return False
        return True