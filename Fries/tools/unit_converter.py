def unit_converter(value, from_unit, to_unit):
    value = float(value)
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()
    aliases = {
        "m": "meter",
        "meters": "meter",
        "km": "kilometer",
        "kilometers": "kilometer",
        "cm": "centimeter",
        "centimeters": "centimeter",
        "mm": "millimeter",
        "millimeters": "millimeter",
        "mi": "mile",
        "miles": "mile",
        "ft": "foot",
        "feet": "foot",
        "in": "inch",
        "inches": "inch",
        "kg": "kilogram",
        "kilograms": "kilogram",
        "g": "gram",
        "grams": "gram",
        "mg": "milligram",
        "milligrams": "milligram",
        "lb": "pound",
        "lbs": "pound",
        "oz": "ounce",
        "c": "celsius",
        "°c": "celsius",
        "f": "fahrenheit",
        "°f": "fahrenheit",
        "k": "kelvin",
        "m/s": "mps",
        "km/h": "kmh",
        "kph": "kmh",
        "mph": "mph",
        "s": "second",
        "sec": "second",
        "min": "minute",
        "h": "hour",
        "hr": "hour",
        "day": "day"
    }
    from_unit = aliases.get(from_unit, from_unit)
    to_unit = aliases.get(to_unit, to_unit)
    length = {
        "meter": 1,
        "kilometer": 1000,
        "centimeter": 0.01,
        "millimeter": 0.001,
        "mile": 1609.344,
        "foot": 0.3048,
        "inch": 0.0254
    }
    if from_unit in length and to_unit in length:
        result = value * length[from_unit] / length[to_unit]
        return {
            "success": True,
            "result": result,
            "from": from_unit,
            "to": to_unit
        }
    mass = {
        "kilogram": 1,
        "gram": 0.001,
        "milligram": 0.000001,
        "pound": 0.45359237,
        "ounce": 0.028349523125
    }
    if from_unit in mass and to_unit in mass:
        result = value * mass[from_unit] / mass[to_unit]
        return {
            "success": True,
            "result": result,
            "from": from_unit,
            "to": to_unit
        }
    temperatures = {"celsius", "fahrenheit", "kelvin"}
    if from_unit in temperatures and to_unit in temperatures:
        if from_unit == "celsius":
            celsius = value
        elif from_unit == "fahrenheit":
            celsius = (value - 32) * 5 / 9
        else:
            celsius = value - 273.15
        if to_unit == "celsius":
            result = celsius
        elif to_unit == "fahrenheit":
            result = (celsius * 9 / 5) + 32
        else:
            result = celsius + 273.15
        return {
            "success": True,
            "result": result,
            "from": from_unit,
            "to": to_unit
        }
    speed = {
        "mps": 1,
        "kmh": 1000 / 3600,
        "mph": 1609.344 / 3600
    }
    if from_unit in speed and to_unit in speed:
        result = value * speed[from_unit] / speed[to_unit]
        return {
            "success": True,
            "result": result,
            "from": from_unit,
            "to": to_unit
        }
    time = {
        "second": 1,
        "minute": 60,
        "hour": 3600,
        "day": 86400
    }
    if from_unit in time and to_unit in time:
        result = value * time[from_unit] / time[to_unit]
        return {
            "success": True,
            "result": result,
            "from": from_unit,
            "to": to_unit
        }
    return {
        "success": False,
        "error": f"Cannot convert {from_unit} to {to_unit}"
    }