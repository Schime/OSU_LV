def bubble_sort_uzlazno(list):
    n = len(list)
    for i in range(n):
        for j in range(0, n-i-1):
            if list[j] > list[j+1]:
                list[j], list[j+1] = list[j+1], list[j]
    return list

def bubble_sort_silazno(list):
    n = len(list)
    for i in range(n):
        for j in range(0, n-i-1):
            if list[j] < list[j+1]:
                list[j], list[j+1] = list[j+1], list[j]
    return list

def aritmeticka_sredina(list):
    if len(list) == 0:
        return 0
    else:
        return sum(list) / len(list)


print("Upišite brojeve: ")
count = 0
numbers = []

while True:
    user_input = input()

    if (user_input == "Done"):
        break
    
    try:
        broj = float(user_input)
        numbers.append(broj)
        count = count + 1
    except ValueError:
        print("Pogrešan unos!")
        continue


print(f"Elementi liste: {numbers}")
print(f"Ukupan broj elemenata liste: {count}")
print(f"Uzlazno sortirana lista: {bubble_sort_uzlazno(numbers)}")
print(f"Silazno sortirana lista: {bubble_sort_silazno(numbers)}")
print(f"Aritmetička sredina liste: {aritmeticka_sredina(numbers)}")