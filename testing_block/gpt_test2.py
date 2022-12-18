import numpy as np

# Create a sequence of words
sentence = "Philip likes to drink wine"
words = sentence.split()

# Create a sequence of word positions
positions = np.arange(len(words))

# Create a sequence of sinusoidal functions with frequency 1, 2, ..., n
round_n  = 3
sinusoids = np.array([[round(np.sin(pos / 10000**(2 * (i // 2) / len(words))),round_n) for i in range(len(words))] for pos in positions])

# Create a sequence of cosinusoidal functions with frequency 1, 2, ..., n
cosinusoids = np.array([[round(np.cos(pos / 10000**(2 * (i // 2) / len(words))),round_n) for i in range(len(words))] for pos in positions])

# Stack the sinusoids and cosinusoids along the second axis to create the final positional encoding
positional_encoding = np.stack([sinusoids, cosinusoids], axis=1)

# Print the final positional encoding
print(positional_encoding)