working_hours = input("Radni sati: ")
hourly_rate = input("eura/h: ")


def total_euro(a, b):
    return a * b


try:
    hours = float(working_hours)
    rate = float(hourly_rate)
    total = total_euro(hours, rate)
    print("Ukupno zarađeno: ", total)
    
except ValueError:
    print("Unesite brojčanu vrijednost za radne sate i plaću po satu.")