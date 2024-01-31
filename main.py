import re
import os
import sys
import nltk
from tqdm import tqdm
import logging
import argparse
import threading
import unicodedata
import numpy as np
import pandas as pd
import language_tool_python

from hunspell import Hunspell
from multiprocessing import Manager
from nltk.tokenize import word_tokenize
from pysentimiento.preprocessing import preprocess_tweet

class SpanishProcessing():
    """
    Spanish Text Pre-Processing class
    """

    def __init__(self, positive_emoji="contento", negative_emoji="triste", hunspell_data_dir="hunspell_dicts/"): 
        """
        Class constructor

        Parameters:
        -----------
        positive_emoji: str
            Positive emoji to be used in the normalization of laughs
        negative_emoji: str
            Negative emoji to be used in the normalization of laughs
        hunspell_data_dir: str
            Path to the hunspell dictionaries
        """

        try:
            self.tool = language_tool_python.LanguageTool('es')  
            self.hunspell = Hunspell('es_ES', hunspell_data_dir=hunspell_data_dir)
        except:
            print("Language tool or hunspell not found")
            raise Exception("Language tool or hunspell not found")
        self.positive_emoji = positive_emoji
        self.negative_emoji = negative_emoji
        self.shared_array = None # multiprocessing Array
        self.dataframe_split = None # Dataframe split by threads
        self.verbose = True # Verbose mode
        
        self.abbreviations = dict() # Abbreviations dictionary
        self.read_abbreviations("abbreviations.txt")
        
    def read_abbreviations(self, file:str) -> None:
        """
        Read abbreviations from file

        Parameters:
        -----------
        file: str
            Path to the file containing the abbreviations
        """
        file = open(file)
        for line in file:
            key, value = line.split()
            value = re.sub(r"(\_)", " ", value)
            self.abbreviations[key] = value

    def lower_case(self, text:str) -> str:
        """
        Lower case the text

        Parameters:
        -----------
        text: str
            Text to be lower cased

        Returns: str
            Lower cased text
        """
        return text.lower()
    
    def remove_non_ascii(self, text:str) -> str:
        """
        Remove non ascii characters

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text without non ascii characters

        Example:
        --------
            Иueve -> ueve
        """
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    def remove_punctuation(self, text:str) -> str:
        """
        Remove punctuation

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text without punctuation
        """

        return re.sub(r'[^\w\s]', '', text)
    
    def remove_numbers(self, text:str) -> str:
        """
        Remove numbers

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text without numbers
        """

        return re.sub(r'\d+', '', text)
    
    def remove_html_tags(self, text:str) -> str:
        """
        Remove html tags

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text without html tags
        
        Example:
        --------
            <p>hola</p> -> hola
        """
        
        return re.sub(r'<.*?>'," ", text)
    
    def remove_extra_spaces(self, text:str) -> str:
        """
        Remove extra spaces

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text without extra spaces
        """

        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_rrrs_elements(self, text:str) -> str:
        """
        Remove rrrs elements (hashtags, mentions, rts and pics)

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text without rrrs elements
        """
        return re.sub(r"#(\w+)"r"|@(\w+)"r"|\b(rts?)\b"r"|\b(pic)\b"," ", text)
    
    def remove_url(self, text:str) -> str:
        """
        Remove urls

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text without urls

        Example:
        --------
            click on https://www.google.com -> click on
        """
        new_text = re.sub(r'https?://\S+|www\.\S+', " ", text)    
        return re.sub(r'(\w+\.(com|es))', " ", new_text)
    
    def normalize_laugh(self, text:str) -> str:
        """
        Normalize laughs

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text with normalized laughs
        
        Example:
        --------
            jajajaja -> contento
        """
        new_text = re.sub(r"(\b(j|a)(j|a)+\b)"r"|(\b(j|e)(j|e)+\b)"r"|(\b(j|i)(j|i)+\b)"r"|(\b(j|o)(j|o)+\b)"r"|(\b(j|u)(j|u)+\b)", self.positive_emoji, text)
        new_text = re.sub(r"(\b(h|a)(h|a)+\b)"r"|(\b(h|e)(h|e)+\b)"r"|(\b(h|i)(h|i)+\b)"r"|(\b(h|o)(h|o)+\b)"r"|(\b(h|u)(h|u)+\b)", self.positive_emoji, new_text)
        new_text = re.sub(r"\b((j+(a|e|i|o|u)+)|((a|e|i|o|u)+j+))((j|a|e|i|o|u)+)\b"r"|\b((h+(a|e|i|o|u)+)|((a|e|i|o|u)+h+))((h|a|e|i|o|u)+)\b", self.positive_emoji, new_text)
        return re.sub(r"\b(x+d+)\b", self.positive_emoji, new_text)   
    
    def normalize_emoticon(self, text:str) -> str:
        """
        Normalize emoticons (get the meaning of the emoticon)

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text with normalized emoticons

        Example:
        --------
            :) -> contento
        """
        new_text = re.sub(r"([\:|\;]+\-*[\)|d]+)", self.positive_emoji, text, re.UNICODE)
        return re.sub(r"([\:|\;]+[\-|\']*\(+)", self.negative_emoji, new_text, re.UNICODE)
    
    def remove_extra_letters(self, text:str) -> str:
        """
        Remove extra letters
        
        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text without extra letters

        Example:
        --------
            hoolaaa -> hola
        """
        return re.sub(r'([a-z])\1{2,}', r'\1', text)        
    
    def normalize_emoji(self, text:str):  
        """
        Normalize emojis (get the meaning of the emoji)

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Text with normalized emojis

        Example:
        --------
            😂 -> cara llorando de risa
        """
   
        tokens = text.split(" ")
        for index, word in enumerate(tokens):
            finded = bool(re.search(r'[^\w\s,.]',word))
            if finded == True:
                try:
                    tokens[index] = preprocess_tweet(word).replace("emoji", "")
                except:
                    pass
        return " ".join(tokens)
    
    def tokenize(self, text:str) -> list:
        """
        Tokenize text (split text into tokens)

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: list
            List of tokens

        Example:
        --------
            hola mundo -> ["hola", "mundo"]
        
        """
        return word_tokenize(text)
    
    def correct_abbreviations(self, tokenized_text:list) -> list:
        """
        Correct abbreviations

        Parameters:
        -----------
        tokenized_text: list
            List of tokens

        Returns: list
            List of tokens with corrected abbreviations

        Example:
        --------
            q -> que
        """
        return [self.abbreviations[word] if word in self.abbreviations else word for word in tokenized_text]

    def correct_spelling_and_grammar(self, tokenized_text:list) -> list:
        """
        Correct spelling and grammar 

        Parameters:
        -----------
        tokenized_text: list
            List of tokens

        Returns: list
            List of tokens with corrected spelling and grammar            
        """

        return [unicodedata.normalize('NFKD', self.tool.correct(word)).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in tokenized_text]
        
    def hunspell_lemmatizer(self, tokenized_text:list) -> list:
        """
        Lemmatize text (get the root of the word)

        Parameters:
        -----------
        tokenized_text: list
            List of tokens

        Returns: list
            List of tokens with lemmatized text

        Example:
        --------
            cantamos -> cantar                     
        """

        return [suggestions[0] if (suggestions := self.hunspell.stem(word)) else word for word in tokenized_text]
    
    def remove_stopwords(self, tokenized_text:list) -> list:
        """
        Remove stopwords (words that do not add meaning to the text)

        Parameters:
        -----------
        tokenized_text: list
            List of tokens

        Returns: list
            List of tokens without stopwords

        Example:
        --------
            hola a todos -> ["hola", "todos"]        
        """
        stopwords = nltk.corpus.stopwords.words('spanish')
        return [i for i in tokenized_text if i not in stopwords]
        

    def preprocessing(self, text:str) -> str:
        """
        Preprocess text 

        Parameters:
        -----------
        text: str
            Text to be processed

        Returns: str
            Preprocessed text

        Note: You can comment or uncomment the functions to be used in the preprocessing
        """
        text = self.lower_case(text)
        text = self.normalize_emoji(text)        
        text = self.remove_non_ascii(text)        
        text = self.remove_html_tags(text)
        text = self.remove_url(text)
        text = self.remove_rrrs_elements(text)
        text = self.remove_numbers(text)        
        text = self.remove_punctuation(text)  
        text = self.remove_extra_letters(text)              
        text = self.remove_extra_spaces(text)        
        
        text = self.normalize_laugh(text)    
        text = self.normalize_emoticon(text)         

        tokenized_text = self.tokenize(text)
        tokenized_text = self.correct_abbreviations(tokenized_text)
        tokenized_text = self.correct_spelling_and_grammar(tokenized_text)
        # tokenized_text = self.remove_stopwords(tokenized_text)
        # tokenized_text = self.hunspell_lemmatizer(tokenized_text)

        text = " ".join(tokenized_text)        
        return text  
    
    def process(self, column:str, thread_id:int) -> None:
        """
        Process function for multiprocessing. Run the preprocessing function of the specified column in the specified thread

        Parameters:
        -----------
        column: str
            Column to be processed
        thread_id: int
            Thread id
        """
        if self.verbose:
            print(f"Thread {thread_id} started")
        tqdm.pandas(desc=f"Processing Thread {thread_id}", leave=False, position=thread_id, mininterval=0.5)
        self.shared_array[thread_id] = self.dataframe_split[thread_id].loc[:,column].progress_apply(lambda x: self.preprocessing(x))     

    def range_process(self, df: pd.DataFrame, column:str, n_threads:int) -> None:
        self.dataframe_split = np.array_split(df, n_threads)
        self.shared_array = Manager().list(range(n_threads))

        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=self.process, args=(column,i))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()

        df["preprocessed"] = pd.concat(self.shared_array, ignore_index=True)
        df.to_csv("preprocessed.csv", index=True, columns=["preprocessed"])

def main(args=None) -> None:
    """
    Main function

    """
    parser = argparse.ArgumentParser(description="Spanish NLP Text Pre-Processing")

    file_option_group = parser.add_mutually_exclusive_group()
    file_option_group.add_argument("-f", "--file", help="CSV Dataset to be processed")
    file_option_group.add_argument("-text", "--text", type=str, help="Text to be processed")

    parser.add_argument("-c", "--column", help="Column to be processed")
    parser.add_argument("-v", "--verbose", type=bool, help="Verbose")
    parser.add_argument("-t", "--threads", type=int, help="Number fo Threads (Activated only if file is specified)")

    args = parser.parse_args()

    sp = SpanishProcessing()
    if not args.verbose:
        sp.verbose = False

    if args.file:
        df = pd.read_csv(args.file)
        if args.column: 
            if isinstance(args.column, int):
                args.column = df.columns[args.column] 

            logging.info(f"Processing, please wait...")
            
            if args.threads:     
                if args.threads > os.cpu_count():
                    args.threads = os.cpu_count()
                    print(f"Threads number must be less than {os.cpu_count()}. Threads number set to {os.cpu_count()}")
                
                else:
                    if args.threads < 1:
                        args.threads = 1
                        print(f"Threads number must be greater than 0. Threads number set to 1")
                    else:
                        if sp.verbose:           
                            print(f"Processing {args.column} column with {args.threads} threads")
                        sp.range_process(df, args.column, args.threads)
            else: 
                tqdm.pandas(desc="Processing", position=0, leave=True)
                df["preprocessed"] = df[args.column].progress_apply(lambda x: sp.preprocessing(x))
            
            df.to_csv("output.csv", index=True, columns=["preprocessed"])
        else:
            print("Column not specified")
            raise Exception("Column not specified")

    elif args.text:        
            print(sp.preprocessing(args.text))
    else:
        raise Exception("File or text not specified")

if __name__ == "__main__":
    tqdm.monitor_interval = 0
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    sys.stdout.flush()
    os.system('cls' if os.name == 'nt' else 'clear')

    #download ntlk data if not downloaded
    if not os.path.exists("nltk_data"):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('words', quiet=True)
        
    try:
        main()
    except Exception as e:
        logging.error(e)
        exit(1)

    