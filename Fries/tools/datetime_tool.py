from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

def time_tool(
    operation,
    date_value=None,
    time_value=None,
    timezone=None,
    target_timezone=None,
    value=None,
    unit=None,
    date_value_2=None,
    time_value_2=None,
    ):
    """
    time and date tool for FlexAI.

    Supported operations:

    current_time
    current_date
    current_day
    current_month
    current_year
    current_hour
    current_minute
    current_second
    current_datetime

    time_in_timezone
    convert_timezone

    day_of_week
    week_number
    days_in_month
    is_leap_year

    date_difference
    time_difference
    days_until
    days_since

    add_time
    subtract_time

    timestamp
    timestamp_to_date
    date_to_timestamp

    format_date
    format_time
    """

    try:
        if operation == "current_time":
            now = datetime.now()
            return {
                "success": True,
                "operation": operation,
                "time": now.strftime("%H:%M:%S"),
                "formatted_time": now.strftime("%I:%M:%S %p"),
            }
        elif operation == "current_date":
            today = date.today()
            return {
                "success": True,
                "operation": operation,
                "date": today.isoformat(),
                "formatted_date": today.strftime("%d %B %Y"),
            }
        elif operation == "current_day":
            today = date.today()
            return {
                "success": True,
                "operation": operation,
                "day": today.strftime("%A"),
                "day_number": today.day,
            }
        elif operation == "current_month":
            today = date.today()
            return {
                "success": True,
                "operation": operation,
                "month": today.strftime("%B"),
                "month_number": today.month,
            }
        elif operation == "current_year":
            today = date.today()
            return {
                "success": True,
                "operation": operation,
                "year": today.year,
            }
        elif operation == "current_hour":
            now = datetime.now()
            return {
                "success": True,
                "operation": operation,
                "hour": now.hour,
            }
        elif operation == "current_minute":
            now = datetime.now()
            return {
                "success": True,
                "operation": operation,
                "minute": now.minute,
            }
        elif operation == "current_second":
            now = datetime.now()
            return {
                "success": True,
                "operation": operation,
                "second": now.second,
            }
        elif operation == "current_datetime":
            now = datetime.now()
            return {
                "success": True,
                "operation": operation,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "formatted": now.strftime("%d %B %Y, %I:%M:%S %p"),
                "day": now.strftime("%A"),
                "month": now.strftime("%B"),
                "year": now.year,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
            }
        elif operation == "time_in_timezone":
            if not timezone:
                return {
                    "success": False,
                    "error": "Timezone is required."
                }
            now = datetime.now(ZoneInfo(timezone))
            return {
                "success": True,
                "operation": operation,
                "timezone": timezone,
                "time": now.strftime("%H:%M:%S"),
                "formatted_time": now.strftime("%I:%M:%S %p"),
                "date": now.strftime("%Y-%m-%d"),
                "formatted_date": now.strftime("%d %B %Y"),
                "day": now.strftime("%A"),
            }
        elif operation == "convert_timezone":
            if not timezone or not target_timezone:
                return {
                    "success": False,
                    "error": "Both source and target timezones are required."
                }
            if not time_value:
                return {
                    "success": False,
                    "error": "time_value is required."
                }
            if date_value:
                dt = datetime.strptime(
                    f"{date_value} {time_value}",
                    "%Y-%m-%d %H:%M:%S"
                )
            else:
                dt = datetime.strptime(
                    time_value,
                    "%H:%M:%S"
                )
                today = date.today()
                dt = datetime.combine(
                    today,
                    dt.time()
                )
            dt = dt.replace(tzinfo=ZoneInfo(timezone))
            converted = dt.astimezone(
                ZoneInfo(target_timezone)
            )
            return {
                "success": True,
                "operation": operation,
                "from_timezone": timezone,
                "to_timezone": target_timezone,
                "date": converted.strftime("%Y-%m-%d"),
                "time": converted.strftime("%H:%M:%S"),
                "formatted": converted.strftime(
                    "%d %B %Y, %I:%M:%S %p"
                ),
                "day": converted.strftime("%A"),
            }
        elif operation == "day_of_week":
            if not date_value:
                return {
                    "success": False,
                    "error": "date_value is required."
                }
            d = datetime.strptime(
                date_value,
                "%Y-%m-%d"
            ).date()
            return {
                "success": True,
                "operation": operation,
                "date": date_value,
                "day": d.strftime("%A"),
            }
        elif operation == "week_number":
            if not date_value:
                return {
                    "success": False,
                    "error": "date_value is required."
                }
            d = datetime.strptime(
                date_value,
                "%Y-%m-%d"
            ).date()
            return {
                "success": True,
                "operation": operation,
                "date": date_value,
                "week_number": d.isocalendar().week,
            }
        elif operation == "days_in_month":
            if not date_value:
                return {
                    "success": False,
                    "error": "date_value is required."
                }
            d = datetime.strptime(
                date_value,
                "%Y-%m-%d"
            ).date()
            if d.month == 12:
                next_month = date(
                    d.year + 1,
                    1,
                    1
                )
            else:
                next_month = date(
                    d.year,
                    d.month + 1,
                    1
                )
            first_day = date(
                d.year,
                d.month,
                1
            )
            days = (next_month - first_day).days
            return {
                "success": True,
                "operation": operation,
                "month": d.strftime("%B"),
                "year": d.year,
                "days": days,
            }
        elif operation == "is_leap_year":
            if value is None:
                return {
                    "success": False,
                    "error": "Year is required."
                }
            year = int(value)
            is_leap = (
                year % 400 == 0
                or (
                    year % 4 == 0
                    and year % 100 != 0
                )
            )
            return {
                "success": True,
                "operation": operation,
                "year": year,
                "is_leap_year": is_leap,
            }
        elif operation == "date_difference":
            if not date_value or not date_value_2:
                return {
                    "success": False,
                    "error": "Two dates are required."
                }
            d1 = datetime.strptime(
                date_value,
                "%Y-%m-%d"
            ).date()
            d2 = datetime.strptime(
                date_value_2,
                "%Y-%m-%d"
            ).date()
            difference = d2 - d1
            total_days = abs(difference.days)
            years = total_days // 365
            remaining_days = total_days % 365
            return {
                "success": True,
                "operation": operation,
                "date_1": date_value,
                "date_2": date_value_2,
                "days": total_days,
                "weeks": total_days // 7,
                "remaining_days_after_weeks": total_days % 7,
                "approx_years": years,
                "approx_remaining_days": remaining_days,
            }
        elif operation == "days_until":
            if not date_value:
                return {
                    "success": False,
                    "error": "Target date is required."
                }
            target = datetime.strptime(
                date_value,
                "%Y-%m-%d"
            ).date()
            today = date.today()
            difference = (
                target - today
            ).days
            return {
                "success": True,
                "operation": operation,
                "today": today.isoformat(),
                "target_date": date_value,
                "days": difference,
            }
        elif operation == "days_since":
            if not date_value:
                return {
                    "success": False,
                    "error": "Date is required."
                }
            past_date = datetime.strptime(
                date_value,
                "%Y-%m-%d"
            ).date()
            today = date.today()
            difference = (
                today - past_date
            ).days
            return {
                "success": True,
                "operation": operation,
                "date": date_value,
                "today": today.isoformat(),
                "days": difference,
            }
        elif operation == "time_difference":
            if not time_value or not time_value_2:
                return {
                    "success": False,
                    "error": "Two times are required."
                }
            t1 = datetime.strptime(
                time_value,
                "%H:%M:%S"
            )
            t2 = datetime.strptime(
                time_value_2,
                "%H:%M:%S"
            )
            difference = abs(
                (t2 - t1).total_seconds()
            )
            hours = int(difference // 3600)
            minutes = int(
                (difference % 3600) // 60
            )
            seconds = int(
                difference % 60
            )
            return {
                "success": True,
                "operation": operation,
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds,
                "total_seconds": int(difference),
            }
        elif operation == "add_time":
            if value is None or unit is None:
                return {
                    "success": False,
                    "error": "value and unit are required."
                }
            now = datetime.now()
            result = add_time_value(
                now,
                float(value),
                unit
            )
            return {
                "success": True,
                "operation": operation,
                "original": now.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "result": result.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        elif operation == "subtract_time":
            if value is None or unit is None:
                return {
                    "success": False,
                    "error": "value and unit are required."
                }
            now = datetime.now()
            result = subtract_time_value(
                now,
                float(value),
                unit
            )
            return {
                "success": True,
                "operation": operation,
                "original": now.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "result": result.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        elif operation == "timestamp":
            now = datetime.now()
            return {
                "success": True,
                "operation": operation,
                "timestamp": int(
                    now.timestamp()
                ),
            }
        elif operation == "timestamp_to_date":
            if value is None:
                return {
                    "success": False,
                    "error": "Timestamp is required."
                }
            timestamp = float(value)
            dt = datetime.fromtimestamp(
                timestamp
            )
            return {
                "success": True,
                "operation": operation,
                "timestamp": timestamp,
                "date": dt.strftime(
                    "%Y-%m-%d"
                ),
                "time": dt.strftime(
                    "%H:%M:%S"
                ),
                "formatted": dt.strftime(
                    "%d %B %Y, %I:%M:%S %p"
                ),
            }
        elif operation == "date_to_timestamp":
            if not date_value:
                return {
                    "success": False,
                    "error": "date_value is required."
                }
            if not time_value:
                time_value = "00:00:00"
            dt = datetime.strptime(
                f"{date_value} {time_value}",
                "%Y-%m-%d %H:%M:%S"
            )
            return {
                "success": True,
                "operation": operation,
                "timestamp": int(
                    dt.timestamp()
                ),
            }

        elif operation == "format_date":
            if not date_value:
                return {
                    "success": False,
                    "error": "date_value is required."
                }
            d = datetime.strptime(
                date_value,
                "%Y-%m-%d"
            )
            return {
                "success": True,
                "operation": operation,
                "original": date_value,
                "formatted": d.strftime(
                    "%A, %d %B %Y"
                ),
            }
        elif operation == "format_time":
            if not time_value:
                return {
                    "success": False,
                    "error": "time_value is required."
                }
            t = datetime.strptime(
                time_value,
                "%H:%M:%S"
            )
            return {
                "success": True,
                "operation": operation,
                "original": time_value,
                "12_hour": t.strftime(
                    "%I:%M:%S %p"
                ),
                "24_hour": t.strftime(
                    "%H:%M:%S"
                ),
            }

        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def add_time_value(dt, value, unit):
    unit = unit.lower()
    if unit in ["second", "seconds", "sec", "secs"]:
        return dt + timedelta(seconds=value)
    elif unit in ["minute", "minutes", "min", "mins"]:
        return dt + timedelta(minutes=value)
    elif unit in ["hour", "hours", "hr", "hrs"]:
        return dt + timedelta(hours=value)
    elif unit in ["day", "days"]:
        return dt + timedelta(days=value)
    elif unit in ["week", "weeks"]:
        return dt + timedelta(weeks=value)
    else:
        raise ValueError(
            f"Unsupported time unit: {unit}"
        )
def subtract_time_value(dt, value, unit):
    unit = unit.lower()
    if unit in ["second", "seconds", "sec", "secs"]:
        return dt - timedelta(seconds=value)
    elif unit in ["minute", "minutes", "min", "mins"]:
        return dt - timedelta(minutes=value)
    elif unit in ["hour", "hours", "hr", "hrs"]:
        return dt - timedelta(hours=value)
    elif unit in ["day", "days"]:
        return dt - timedelta(days=value)
    elif unit in ["week", "weeks"]:
        return dt - timedelta(weeks=value)
    else:
        raise ValueError(
            f"Unsupported time unit: {unit}"
        )

if __name__ == "__main__":
    print(
        time_tool(
            "current_datetime"
        )
    )
    print(
        time_tool(
            "current_time"
        )
    )
    print(
        time_tool(
            "current_date"
        )
    )
    print(
        time_tool(
            "current_day"
        )
    )
    print(
        time_tool(
            "current_month"
        )
    )
    print(
        time_tool(
            "current_year"
        )
    )