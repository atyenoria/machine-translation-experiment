from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
chinese_text = "生活就像一盒巧克力。"
ja_text = "「国民年金」は原則、全ての国民が将来的にもらえ、今年度は月額約6万5000円が受け取れます。満額を受給するためには、現在の制度では20歳から59歳までの40年間、保険料を払い続けなければいけません。厚労省の部会では、これをさらに5年延長し、64歳までの45年間にするか検討される見通しです。"
en_text="A polarizing figure in Japanese politics, Abe's supporters described him as a patriot who worked to strengthen Japan's security and international stature, while his opponents described his nationalistic policies and negationist views on history as threatening Japanese pacifism and damaging relations with East Asian neighbors China and South Korea. Commentators have said that his legacy pushed Japan towards more proactive military spending, security, and economic policies."

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# translate Hindi to French
tokenizer.src_lang = "hi"
encoded_hi = tokenizer(hi_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("fr"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "La vie est comme une boîte de chocolat."

# translate Chinese to English
tokenizer.src_lang = "zh"
encoded_zh = tokenizer(chinese_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Life is like a box of chocolate."



tokenizer.src_lang = "ja"
encoded_zh = tokenizer(ja_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


tokenizer.src_lang = "en"
encoded_zh = tokenizer(en_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("ja"))
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))