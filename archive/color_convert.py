import re

# Paste your raw Swift CIColor lines here
swift_colors = """
CIColor(red: 0.114, green: 0.259, blue: 0.537)
CIColor(red: 0, green: 0.749, blue: 0.698)
CIColor(red: 0.827, green: 0.153, blue: 0.243)
CIColor(red: 0.863, green: 0.345, blue: 0.165)
CIColor(red: 1.0, green: 0.784, blue: 0.271)
CIColor(red: 0.420, green: 0.557, blue: 0.137)
CIColor(red: 0.596, green: 0.984, blue: 0.596)
"""

# Extract numbers using regex
pattern = r"red:\s*([\d.]+),\s*green:\s*([\d.]+),\s*blue:\s*([\d.]+)"
matches = re.findall(pattern, swift_colors)

for match in matches:
    r, g, b = map(float, match)
    rgb = tuple(round(c * 255) for c in (r, g, b))
    # print(f"Normalized: ({r}, {g}, {b}) -> RGB 255: {rgb}")
    print(f"{[rgb[0], rgb[1], rgb[2]],}")