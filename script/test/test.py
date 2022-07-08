import cv2
import json

img = cv2.imread("/Users/tinglyfeng/Desktop/dep_deploy/MNN/resource/images/cat.jpg")
print()

res = {}
with open("/Users/tinglyfeng/Desktop/classes.txt") as f:
    data = f.readlines()

outline = ""
for line in data:
    line = line.strip()
    line = line.replace("\'", "")
    line = line.replace("\"", "")
    index = int(line.split('\t')[0])
    className = line.split('\t')[1]
    outline += "{" + str(index) + "," + "\""+className + "\"" + "}" + ",\\" + "\n"

    res[index] = className
print(outline)

with open('result.json', 'w') as f:
    json.dump(res, f)

print()