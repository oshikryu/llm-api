class HelloWorld:
    def __init__(self, message="Hello, World!"):
        self.message = message

    def print_message(self):
        print(self.message)

if __name__ == "__main__":
    hello = HelloWorld()
    hello.print_message()
