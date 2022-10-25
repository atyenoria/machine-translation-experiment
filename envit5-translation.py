from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# https://huggingface.co/VietAI/envit5-translation?text=This+is+a+test

model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cpu")
model.to(device)
inputs = [
    "vi: VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo và xây dựng một cộng đồng các chuyên gia trong lĩnh vực trí tuệ nhân tạo đẳng cấp quốc tế tại Việt Nam.",
    "vi: Theo báo cáo mới nhất của Linkedin về danh sách việc làm triển vọng với mức lương hấp dẫn năm 2020, các chức danh công việc liên quan đến AI như Chuyên gia AI (Artificial Intelligence Specialist), Kỹ sư ML (Machine Learning Engineer) đều xếp thứ hạng cao.",
    "en: Our teams aspire to make discoveries that impact everyone, and core to our approach is sharing our research and tools to fuel progress in the field.",
    "en: We're on a journey to advance and democratize artificial intelligence through open source and open science."
    ]

tokenized_text = tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to(device)

outputs = model.generate(tokenized_text, max_length=512).to(device)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

