import requests

url = "https://shengwen.goodsop.cn:30006/asr"
# url = "http://182.148.54.70:9006/asr"
audio_file = "/home/dieu/project/asr_0928_v1.1/新录音.wav"  # 替换为你的音频路径

with open(audio_file, "rb") as f:
    files = {"audio": (audio_file, f, "audio/wav")}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())