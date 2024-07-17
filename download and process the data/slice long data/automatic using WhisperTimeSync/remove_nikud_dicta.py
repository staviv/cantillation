import requests

def remove_nikud_dicta(text, maleify=False):
  """
  Removes nikud from Hebrew text using Dicta's API.

  Args:
    text: Hebrew text with nikud.
    maleify: Boolean indicating whether to add maleify (אימות קריאה) or not.

  Returns:
    Hebrew text without nikud, or an error message if an error occurred.
  """
  api_endpoint = 'https://remove-nikud-2-0.loadbalancer2.dicta.org.il/api'
  payload = {
    "data": text,
    "genre": "rabbinic", 
    "fQQ": True,
    "maleify": ~maleify,  # Add maleify parameter to the payload
    "dasda": True
  }
  
  try:
    response = requests.post(api_endpoint, json=payload)
    response.raise_for_status()
    cleaned_text = response.json()['results'].replace('\u05BD', '') 
    return cleaned_text
  except requests.exceptions.RequestException as e:
    return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
  text = "בְנֵי־הָֽאֱלֹהִים֙ אֶת־בְּנ֣וֹת הָֽאָדָ֔ם כִּ֥י טֹבֹ֖ת הֵ֑נָּה וַיִּקְח֤וּ לָהֶם֙ "

  # Remove nikud without maleify
  cleaned_text_no_maleify = remove_nikud_dicta(text)
  print("Without Maleify:", cleaned_text_no_maleify)

  # Remove nikud with maleify
  cleaned_text_with_maleify = remove_nikud_dicta(text, maleify=True)
  print("With Maleify:", cleaned_text_with_maleify)