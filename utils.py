import ast

def extract_names(x):
    """
    Safely extracts and concatenates 'name' fields from a string representation of a list of dictionaries.

    Args:
        x (str): A string formatted like a list of dictionaries, e.g., 
                 "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Adventure'}]".

    Returns:
        str: A space-separated string of names with internal spaces removed (e.g., "Action Adventure" → "ActionAdventure").
             Returns an empty string if input is invalid or parsing fails.

    Example:
        >>> extract_names("[{'id': 1, 'name': 'Science Fiction'}, {'id': 2, 'name': 'Drama'}]")
        'ScienceFiction Drama'
    """
    try:
        return " ".join([i['name'].replace(" ", "") for i in ast.literal_eval(x)])
    except:
        return ""
