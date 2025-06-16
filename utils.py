import ast

def extract_names(x):
    try:
        return " ".join([i['name'].replace(" ", "") for i in ast.literal_eval(x)])
    except:
        return ""
