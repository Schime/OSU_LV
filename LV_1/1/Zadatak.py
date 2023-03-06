working_hours = float(input("Radni sati: "))

hourly_rate = float(input("eura/h: "))


def total_euro(a, b):
    return a * b


result = total_euro(working_hours, hourly_rate)

print(f"Ukupno: {result} eura")