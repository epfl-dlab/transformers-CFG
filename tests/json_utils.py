import json


def is_json_parsable(string):
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False
    except Exception as e:
        # You might want to handle or log other exceptions as well
        return False
