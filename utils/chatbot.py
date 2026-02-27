def medical_chatbot(user_input):
    if "fever" in user_input.lower():
        return "Drink fluids and rest. If fever persists, consult doctor."
    elif "chest pain" in user_input.lower():
        return "Please seek immediate medical attention."
    else:
        return "Please consult a healthcare professional."
