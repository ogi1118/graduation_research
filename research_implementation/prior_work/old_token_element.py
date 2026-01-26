
class TokenElement:
    def __init__(self, token_string, vector) -> None:
        self.token_string = token_string
        self.vector = vector

    def get_token_string(self):
        return self.token_string
    
    def get_vector(self):
        return self.vector