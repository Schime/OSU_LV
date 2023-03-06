with open('song.txt', 'r') as file:
    word_count = {}
    for line in file:
        words = line.split()  # razdvoji liniju na riječi
        for word in words:
            # ignoriraj znakove interpunkcije i pretvori sve riječi u mala slova
            word = word.lower().strip(',.!?')   # strip miče sve zareze, točke, uskličnike i upitnike
            if word.isalpha():  # brojimo samo riječi koje se sastoje od slova
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

# ispis
unique_words = [word for word, count in word_count.items() if count == 1]
print("Broj riječi koje se pojavljuju samo jednom:", len(unique_words))
print("Te riječi su:", unique_words)