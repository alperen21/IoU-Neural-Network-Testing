class Config:
    def __init__(self, input_dir="input", models_dir="models", delimiter=", ", threshold=0.5) -> None:
        self.input_dir = input_dir
        self.models_dir = models_dir
        self.delimiter = delimiter
        self.threshold = threshold

config = Config()