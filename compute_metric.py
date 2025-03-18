import os
import json

# 将合并后的数据写入目标文件
with open("result.json", "r", encoding="utf-8") as f:
    result = json.load(f)

correct = 0
n = len(result)
for data in result:
    if data["answer"] == data["model_pred"]:
        correct += 1


print(f"Accuracy: {correct / n}")

