import os
import sys

import wave
import pyaudio

RECORD_FORMAT = pyaudio.paInt16
RECORD_RATE = 44100
RECORD_CHANNELS = 1
RECORD_CHUNK = 1024
recoder = pyaudio.PyAudio()

RECORD_SECONDS = 1
FRAME_LENGTH = int(RECORD_RATE / RECORD_CHUNK * RECORD_SECONDS)

CACHE_FOLDER = os.path.join(".", "cache")
OUTPUT_FILENAME = "conversation_output.wav"

if not os.path.isdir(CACHE_FOLDER):
    os.mkdir(CACHE_FOLDER)


def print_interrupt_message():
    print("""
-------------------------------------------
INFO: Recording interrupted.
-------------------------------------------""", file=sys.stderr)

def record_audio(params):
    stream = recoder.open(
        format=RECORD_FORMAT, channels=RECORD_CHANNELS,
        rate=RECORD_RATE, input=True,
        frames_per_buffer=RECORD_CHUNK
    )

    print("""
-------------------------------------------
INFO: Recording started...
-------------------------------------------
""")

    output_file = wave.open(OUTPUT_FILENAME, "wb")
    output_file.setnchannels(RECORD_CHANNELS)
    output_file.setsampwidth(recoder.get_sample_size(RECORD_FORMAT))
    output_file.setframerate(RECORD_RATE)

    try:
        while True:
            if params['interrupted']:
                raise KeyboardInterrupt

            read = [stream.read(RECORD_CHUNK) for _ in range(FRAME_LENGTH)]
            frame = b"".join(read)
            output_file.writeframes(frame)
            params['duration'] += len(read) / RECORD_RATE * RECORD_CHUNK
    except BrokenPipeError:
        pass
    except KeyboardInterrupt:
        pass

    print_interrupt_message()

    stream.stop_stream()
    stream.close()
    output_file.close()
