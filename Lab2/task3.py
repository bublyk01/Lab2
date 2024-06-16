import numpy as np


def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    encrypted_vector = np.dot(key_matrix, message_vector)
    return encrypted_vector


def decrypt_message(encrypted_vector, key_matrix):
    inv_key_matrix = np.linalg.inv(key_matrix)
    decrypted_vector = np.dot(inv_key_matrix, encrypted_vector)
    decrypted_message = ''.join([chr(int(round(num))) for num in decrypted_vector])
    return decrypted_message

message = "Hello, World!"
message_length = len(message)

key_matrix = np.random.randint(1, 10, size=(message_length, message_length))

while np.linalg.det(key_matrix) == 0:
    key_matrix = np.random.randint(1, 10, size=(message_length, message_length))

encrypted_vector = encrypt_message(message, key_matrix)
print("Encrypted vector:", encrypted_vector)

decrypted_message = decrypt_message(encrypted_vector, key_matrix)
print("Decrypted message:", decrypted_message)
