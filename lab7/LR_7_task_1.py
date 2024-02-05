from data import distances, start_city, cities
from ant_colony import AntColony
import matplotlib.pyplot as plt

ant_colony = AntColony(distances, 30, 20, 500, 0.8)

result = ant_colony.run(start=start_city)
print(f"Найкоротший шлях: {result[1]} км")

# Знайдений шлях
path = "Шлях: "
for i in result[0]:
    path += f"{cities[i[0]]} -> "
print(path[:-4])

# Графік для найкоротшого маршруту
fig = plt.figure(figsize=(10, 10))

plt.xticks([i + 1 for i in range(len(cities))])
plt.yticks([i for i in range(len(cities))], cities)

plt.xlabel("Номери міст")
plt.ylabel("Назви міст")

plt.title("Маршрут коміявожера")

plt.plot([i + 1 for i in range(len(result[0]))], [i[0] for i in result[0]], ms=12, marker='o', mfc='b', mew=4,
         color='#FF5733', linestyle='--')

plt.grid()
plt.show()
