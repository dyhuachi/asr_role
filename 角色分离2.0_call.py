import requests

# url = "http://182.148.54.70:9006/transcribe"
# url = "http://localhost:8002/transcribe"
url = "https://shengwen.goodsop.cn:30006/transcribe"
# url = "http://172.16.20.52:8002/transcribe"
files = {
    "audio": open("/home/dieu/下载/session-98319091-8fbf-450c-a9aa-650120edad1b_1.wav", "rb"),
    # "audio": open("/home/dieu/下载/gq_jwj.wav", "rb"),
    "doctor_enroll": open("/home/dieu/下载/session-e4c2b6af-87de-4137-9642-114020111cd8.wav", "rb")
}
import time
a = time.time()
response = requests.post(url, files=files)
b = time.time()
print(f"耗时：{b-a}s")
print(response.json())


# import requests
#
# # 替换为你的实际 URL
# MAIN_URL = "https://devfiles.goodsop.cn/files/20251225/938823/data_test/2003760586345164801/mz/session-b4d68193-dae6-4189-8c2c-a3e0e9cf87e2_1.wav"
# DOCTOR_URL = "https://devfiles.goodsop.cn/files/20251224/938823/data_test/2003760586345164801/sw/session-e4c2b6af-87de-4137-9642-114020111cd8.wav"  # 必须是真实有效的医生音频
#
# # 调用 transcribe
# resp = requests.post(
#     "http://localhost:8002/transcribe",
#     json={
#         "audio_url": MAIN_URL,
#         "doctor_enroll_url": DOCTOR_URL
#     }
# )
#
# print("Status:", resp.status_code)
# print("Result:", resp.json())