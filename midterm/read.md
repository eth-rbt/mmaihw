Midterm baseline evaluation: 

Which model+resolution: Qwen3-VL-4B (4bit)
https://huggingface.co/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit?utm

Which eval: Fine-grain why recognition https://github.com/hd-epic/hd-epic-annotations/blob/main/vqa-benchmark/fine_grained_why_recognition.json


How long of the video we feed: (32 seconds)
Context window:  16s before + 16s after
Prompt: Directly from the VQA benchmark
FPS: 1fps
Modality: Picture frames + (text)
Temperature: 0.0
Evaluation: Correct/Wrong
How to tokenize the video: 

System: 
RTX 3070, windows computer

Data path: 
For testing: 
in hdepic_example, file P01-20240202-110250
For actual eval running:
in hdpic_data, from forlder P1 to P9, within each foler there is some mp4 files (not the full set) 