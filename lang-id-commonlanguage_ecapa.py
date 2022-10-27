import torchaudio
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
# Italian Example
out_prob, score, index, text_lab = classifier.classify_file('speechbrain/lang-id-commonlanguage_ecapa/example-it.wav')
print(text_lab)

# French Example
out_prob, score, index, text_lab = classifier.classify_file('speechbrain/lang-id-commonlanguage_ecapa/example-fr.wav')
print(text_lab)