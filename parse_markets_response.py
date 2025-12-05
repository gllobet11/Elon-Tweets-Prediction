import json

file_path = "markets_response.json"

# Read the content of the file
with open(file_path, "r") as f:
    content = f.read()

# Extract the JSON part from the PowerShell output
json_start = content.find("Content           : {")
if json_start != -1:
    json_str = content[json_start + len("Content           : ") :].strip()
    # Try to clean up any trailing characters from PowerShell output
    if json_str.endswith("..."):
        json_str = json_str[:-3] + '"}]}'  # Heuristic: fix truncated JSON

    try:
        data = json.loads(json_str)
        if data and "data" in data and len(data["data"]) > 0:
            first_market = data["data"][0]
            asset_id = first_market.get("asset_id") or first_market.get(
                "market"
            )  # Check for both
            if asset_id:
                print(f"Found asset_id: {asset_id}")
            else:
                print(
                    "Could not find 'asset_id' or 'market' in the first market object."
                )
        else:
            print("No market data found in the JSON response.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(
            f"Attempted to decode: {json_str[:500]}..."
        )  # Print beginning of string for debug
else:
    print("Could not find JSON content in the file.")
