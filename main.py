# import librosa
# import tensorflow as tf
# import soundfile as sf
# import tensorflow_hub as hub
# import numpy as np
# from tqdm import tqdm


# def denoise_audio(audio, sample_rate):
#     try:
#         # Resample the audio to 16000 Hz
#         resampled_audio = librosa.resample(
#             audio, orig_sr=sample_rate, target_sr=16000)

#         # Convert the resampled audio to a tensor
#         resampled_audio_tensor = tf.convert_to_tensor(
#             resampled_audio, dtype=tf.float32)

#         # Reshape the tensor to have a batch size of 1
#         resampled_audio_tensor = tf.reshape(resampled_audio_tensor, (1, -1))

#         # Load the denoising model from TensorFlow Hub
#         model_url = "https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1"
#         model = hub.load(model_url)

#         # Denoise the audio
#         denoised_audio_tensor = model(resampled_audio_tensor, True, None)
#         denoised_audio = denoised_audio_tensor['embedding'].numpy()

#         # Convert the denoised audio to a NumPy array
#         denoised_audio = denoised_audio.squeeze() / np.max(np.abs(denoised_audio))

#         return denoised_audio

#     except Exception as e:
#         print(f"Error occurred while denoising audio: {str(e)}")
#         return None


# def main():
#     audio_file = 'test.wav'
#     audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)

#     output_file = 'denoised_audio.wav'
#     with sf.SoundFile(output_file, mode='w', samplerate=sample_rate, channels=1, subtype='PCM_16') as file:
#         for i in tqdm(range(0, len(audio), sample_rate), desc="Processing audio"):
#             chunk = audio[i:i+sample_rate]
#             denoised_chunk = denoise_audio(chunk, sample_rate)

#             if denoised_chunk is not None:
#                 file.write(denoised_chunk)


# if __name__ == '__main__':
#     main()


import librosa
import tensorflow as tf
import soundfile as sf
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def denoise_chunk(chunk, sample_rate):
    try:
        # Resample the audio to 16000 Hz
        resampled_audio = librosa.resample(
            chunk, orig_sr=sample_rate, target_sr=16000)

        # Convert the resampled audio to a tensor
        resampled_audio_tensor = tf.convert_to_tensor(
            resampled_audio, dtype=tf.float32)

        # Reshape the tensor to have a batch size of 1
        resampled_audio_tensor = tf.reshape(resampled_audio_tensor, (1, -1))

        # Load the denoising model from TensorFlow Hub
        model_url = "https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1"
        model = hub.load(model_url)

        # Denoise the audio
        denoised_audio_tensor = model(resampled_audio_tensor, True, None)
        denoised_audio = denoised_audio_tensor['embedding'].numpy()

        # Convert the denoised audio to a NumPy array
        denoised_audio = denoised_audio.squeeze() / np.max(np.abs(denoised_audio))

        return denoised_audio

    except Exception as e:
        print(f"Error occurred while denoising audio: {str(e)}")
        return None


def main():
    audio_file = 'test.wav'
    audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)

    output_file = 'denoised_audio.wav'
    with sf.SoundFile(output_file, mode='w', samplerate=sample_rate, channels=1, subtype='PCM_16') as file:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in tqdm(range(0, len(audio), sample_rate), desc="Processing audio"):
                chunk = audio[i:i+sample_rate]
                futures.append(executor.submit(
                    denoise_chunk, chunk, sample_rate))

            for future in tqdm(futures, desc="Writing audio"):
                denoised_chunk = future.result()
                if denoised_chunk is not None:
                    file.write(denoised_chunk)


if __name__ == '__main__':
    main()
