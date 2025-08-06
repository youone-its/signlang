from collections import Counter

def load_allowed_keywords(file_path='model/labels.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f if line.strip())

# Deteksi dan hapus frasa berulang (n-gram size 2 sampai 3)
def remove_repeated_phrases(words, max_ngram=3):
    i = 0
    result = []

    while i < len(words):
        found_repeat = False

        # Coba n-gram 3 ke bawah
        for n in range(max_ngram, 0, -1):
            if i + 2 * n <= len(words) and words[i:i+n] == words[i+n:i+2*n]:
                # Frasa n-gram diulang → ambil sekali
                result += words[i:i+n]
                i += 2 * n
                found_repeat = True
                break
        
        if not found_repeat:
            result.append(words[i])
            i += 1

    return result


def clean_out_of_context(words, allowed_keywords):
    # Filter keyword valid
    words = [w for w in words if w in allowed_keywords]

    # Step 1: hapus frasa berulang
    words = remove_repeated_phrases(words)

    # Step 2: Filter kata berdasarkan konteks
    # Kasus khusus untuk terimakasih
    if "terimakasih" in words:
        idx = words.index("terimakasih")
        if idx < len(words) - 1:  # terimakasih bukan di akhir kalimat
            words.remove("terimakasih")
        elif idx > 0 and words[idx - 1] not in {"saya", "kami"}:
            # Tidak diikuti oleh "saya berterimakasih", buang juga
            words.remove("terimakasih")

    return ' '.join(words)

def postprocess_gesture_output(raw_text, label_file='model/labels.txt'):
    allowed_keywords = load_allowed_keywords(label_file)
    raw_words = raw_text.strip().split()
    final_result = clean_out_of_context(raw_words, allowed_keywords)
    return final_result

# Tes
if __name__ == "__main__":
    raw_input = "perkenalkan nama perkenalkan nama perkenalkan terimakasih saya"
    cleaned = postprocess_gesture_output(raw_input)
    print("🧹 Output bersih:", cleaned)
