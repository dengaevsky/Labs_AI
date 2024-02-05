import numpy as np
import neurolab as nl

# Оновлені тренувальні та тестові дані для літер Б, О, В
target = np.array([[1, 1, 1, 1, 1,
                    1, 0, 0, 0, 0,
                    1, 1, 1, 1, 1,  # Б
                    1, 0, 0, 0, 1,
                    1, 1, 1, 1, 1],

                   [0, 1, 0, 1, 0,
                    1, 0, 0, 0, 1,
                    1, 0, 1, 0, 0,
                    1, 0, 0, 1, 1,
                    0, 1, 1, 1, 0],  # О

                   [1, 1, 1, 1, 0,
                    1, 0, 0, 0, 1,
                    1, 1, 1, 1, 0,  # В
                    1, 0, 0, 0, 1,
                    1, 1, 1, 1, 0]])

target_defaced = np.array([[1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1,  # Б
                            1, 0, 0, 1, 1,
                            1, 1, 1, 1, 1],

                           [0, 1, 0, 1, 0,
                            1, 0, 0, 0, 1,
                            1, 1, 1, 0, 0,
                            1, 0, 0, 1, 1,
                            0, 1, 0, 1, 0],  # О

                           [1, 1, 1, 1, 0,
                            1, 0, 0, 0, 1,
                            1, 1, 1, 1, 0,  # В
                            1, 0, 0, 0, 1,
                            1, 0, 1, 1, 0]])

chars = ['Г', 'Д', 'О']
target[target == 0] = -1

# Створення та тренування мережі
net = nl.net.newhop(target.reshape(-1, 25))  # Плоске представлення для мережі Хопфілда
output = net.sim(target.reshape(-1, 25))
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())


# Тестування нейронної мережі для всіх трьох літер
def test_defaced_letter(letter, target, net):
    test = np.asfarray(letter).flatten()
    test[test == 0] = -1
    output = net.sim([test])
    result = (output[0] == target.flatten()).all()
    steps = len(net.layers[0].outs)
    return result, steps


target_letters = [target[0], target[1], target[2]]

# Тестування для літер Г, Д, О
for i in range(len(target_letters)):
    print(f"\nTest on defaced {chars[i]}:")
    result, steps = test_defaced_letter(target_letters[i], target_letters[i], net)
    print(result, 'Sim. steps', steps)
