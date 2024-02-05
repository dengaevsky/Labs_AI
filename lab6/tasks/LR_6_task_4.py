import numpy as np
import neurolab as nl

target = [[1, 0, 0, 0, 1,
           1, 1, 0, 0, 1,
           1, 0, 1, 0, 1,
           1, 0, 0, 1, 1,
           1, 0, 0, 0, 1],
          [1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 0,
           1, 0, 0, 1, 0,
           1, 0, 0, 0, 1],
          [0, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           0, 1, 1, 1, 0]]

chars = ['N', 'E', 'R', 'O']
target = np.asfarray(target)
target[target == 0] = -1

# Створення та тренування мережі
net = nl.net.newhop(target)
output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())


# Тестування нейронної мережі
def test_defaced_letter(letter, target, net):
    test = np.asfarray(letter)
    test[test == 0] = -1
    output = net.sim([test])
    result = (output[0] == target).all()
    steps = len(net.layers[0].outs)
    return result, steps

target_letters = [target[0], target[1], target[2], target[3]]

print("\nTest on defaced N:")
result, steps = test_defaced_letter([0, 0, 0, 0, 0,
                                     1, 1, 0, 0, 1,
                                     1, 1, 0, 0, 1,
                                     1, 0, 1, 1, 1,
                                     0, 0, 0, 1, 1], target_letters[0], net)
print(result, 'Sim. steps', steps)

print("\nTest on defaced E:")
result, steps = test_defaced_letter([0, 0, 0, 0, 0,
                                     0, 1, 1, 1, 1,
                                     0, 1, 1, 1, 1,
                                     0, 1, 1, 1, 1,
                                     0, 0, 0, 0, 0], target_letters[1], net)
print(result, 'Sim. steps', steps)

print("\nTest of defaced R:")
result, steps = test_defaced_letter([1, 1, 0, 0, 0,
                                     1, 0, 0, 0, 1,
                                     1, 1, 1, 1, 0,
                                     0, 1, 0, 1, 0,
                                     1, 0, 0, 0, 1], target_letters[2], net)
print(result, 'Sim. steps', steps)

print("\nTest of defaced O:")
result, steps = test_defaced_letter([0, 1, 1, 1, 0,
                                     1, 0, 0, 0, 1,
                                     0, 0, 1, 0, 1,
                                     1, 0, 0, 0, 1,
                                     0, 1, 0, 1, 0], target_letters[3], net)
print(result, 'Sim. steps', steps)