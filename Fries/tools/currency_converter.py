import requests

def currency_converter(amount, from_currency, to_currency):
    amount = float(amount)
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()
    url = (
        "https://api.frankfurter.app/latest"
        f"?amount={amount}"
        f"&from={from_currency}"
        f"&to={to_currency}"
    )
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Currency API error: {response.status_code}"
            }
        data = response.json()
        if "rates" not in data:
            return {
                "success": False,
                "error": "Could not get exchange rate."
            }
        result = data["rates"].get(to_currency)
        if result is None:
            return {
                "success": False,
                "error": f"Unknown currency: {to_currency}"
            }
        return {
            "success": True,
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "result": result,
            "date": data.get("date")
        }
    except requests.RequestException as e:
        return {
            "success": False,
            "error": str(e)
        }