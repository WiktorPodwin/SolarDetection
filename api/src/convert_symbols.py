import logging

class ConvertSymbols:
    """
    Class for converting symbols
    """
    def slash_into_dash(self, word) -> str:
        """
        Converts every slash into dash in word
        
        Args:
            word: Word to convert
        """
        try:
            new_word = ""
            for symbol in str(word):
                if symbol == "/":
                    new_word += "-"
                else:
                    new_word += symbol
            print(new_word)
            return new_word
        except TypeError as e:
            logging.error("Error while converting slash into dash: %s", e)
            return word
    
    def dash_into_slash(self, word: str) -> str:
        """
        Converts every dash into slash in word
        
        Args:
            word: Word to convert
        """
        try:
            new_word = ""
            for symbol in word:
                if symbol == "-":
                    new_word += "/"
                else:
                    new_word += symbol
            return new_word
        except TypeError as e:
            logging.error("Error while converting dash into slash: %s", e)
            return word
        