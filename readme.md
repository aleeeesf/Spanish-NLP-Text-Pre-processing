# Spanish Text Preprocessing for NLP

This repository provides a Spanish text preprocessing tool that allows efficient cleaning and transformation of data, including the capability to process text and CSV files in parallel using multiple threads.

## Features

- Text preprocessing in Spanish.
- Efficient processing of CSV files in parallel.
- Support for specifying the number of threads and column when processing CSV files.
- Easy-to-use with command-line options.

## Dependencies

Make sure to have the dependencies specified in requirements.txt installed. You can install them using:

```bash
pip install -r requirements.txt
```

## Usage
### Text preprocessing

To preprocess text, use the following option:

```bash
python preprocess.py -text "Tu texto aquí"
```

### CSV File Preprocessing

To preprocess a CSV file, use the following option:

```bash
python preprocess.py -file "<filename>" -t <number_of_threads> -c <column_name> -v <boolean>
```

* <filename>: CSV file to process.
* <number_of_threads>: Number of threads for parallel processing.
* <column_name>: Name of the column to process in the CSV file.
* <boolean>: If False, the output won´t be shown in the console. 


### Preprocessing options

In the preprocess function, users can comment or uncomment any function according to their specific needs. By default, lemmatization and stopword removal are commented. This allows users to customize text preprocessing according to their preferences.

```bash
# In source code:
# Uncomment or comment functions based on user preferences

def preprocess(text):
    # Uncomment or comment functions based on user preferences
    def preprocessing(self, text:str) -> str:
        text = self.lower_case(text)
        text = self.normalize_emoji(text)        
        text = self.remove_non_ascii(text)        
        ....

        tokenized_text = self.tokenize(text)
        # tokenized_text = self.remove_stopwords(tokenized_text)
        # tokenized_text = self.hunspell_lemmatizer(tokenized_text)

        text = " ".join(tokenized_text)        
        return text 
```



#### Examples:

```bash
python script.py -f "archivo.csv" -c "text" -v false -t 4
```

```bash
python script.py -text "Texto a procesar"
```

#### Preprocess examples:

1. "Ayer fui a ber a mi familia y nos sacamos fotos en el parqe." -> "ayer fui a ver a mi familia y nos sacamos fotos en el parque"
2. "Me compre un coxe nuebo, esta divino! 🚗✨" -> "me compre un come nuevo esta divino coche chispas"