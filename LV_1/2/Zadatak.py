try:
    input = float(input("Upisite broj u rasponu [0.0 1.0]: "))

    if input < 0.0 or input > 1.0:
        print("Greška: broj nije unutar intervala!")

    elif(input < 0.6):
        print("F")

    elif(0.6 <= input < 0.7):
        print("D")

    elif(0.7 <= input < 0.8):
        print("C")

    elif(0.8 <= input < 0.9):
        print("B")

    elif(input >= 0.9):
        print("A")

except ValueError:
    print("Greška!")