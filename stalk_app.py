from multiprocessing import Process, Manager
#from threading import Thread
from time import sleep
import os
import gc

import torch
import torchaudio
from pyannote.core import Segment


from stalk_streamer import recoder, record_audio, CACHE_FOLDER, OUTPUT_FILENAME


if __name__ == '__main__':
    with Manager() as manager:
        from stalk_models import ChatHistory, llama3, whisper, audio, resnet152, system_prompt, token_stream

        speaker1 = "민서", "./SpeakerDiarization/sample_conversation/real/sentence_F.wav"
        speaker2 = "연우", "./SpeakerDiarization/sample_conversation/real/sentence_M.wav"
        resnet152.register(*speaker1)
        resnet152.register(*speaker2)

        user_prompt = f"Based on the conversations between {speaker1[0]} and {speaker2[0]}, on be half of {speaker2[0]}, do recommend a new topic sentence related the current situation or their personal interests."

        def get_recommendation():
            for chunk in llama3(system_prompt, user_prompt):
                print(token_stream(chunk), end="", flush=True)
            print()

        input("\n\nInitialization finished! Press Enter to start conversation...")
        print()

        RECORD_PARAMS = manager.dict(interrupted=False, duration=0.0)

        record_thread = Process(target=record_audio, kwargs=dict(params=RECORD_PARAMS))
        record_thread.start()

        start_offset = 0.0
        temp_file = os.path.join(CACHE_FOLDER, "temp.wav")

        void_count = 0
        error_count = 0

        try:
            while not RECORD_PARAMS['duration']:
                sleep(0.001)  # Wait until the recording starts

            while record_thread.is_alive():
                audio_range = Segment(start_offset, RECORD_PARAMS['duration'])
                #print("Transcribing audio...", audio_range)
                torchaudio.save(temp_file, *audio.crop(OUTPUT_FILENAME, audio_range))

                segments, info = whisper(temp_file, beam_size=5, word_timestamps=False)
                #print("Transcription finished.")
                segments = iter(segments)

                for segment in segments:
                    try:
                        crop_range = (start_offset + segment.start, start_offset + segment.end)
                        portion = audio.crop(OUTPUT_FILENAME, Segment(crop_range[0], crop_range[1]))
                        torchaudio.save(os.path.join(CACHE_FOLDER, f"{crop_range[0]:.2f}.wav"), *portion)

                        speaker = resnet152(*portion)

                        if start_offset != crop_range[0]:
                            print()
                        print(f"\r{crop_range} -> [{speaker['name']}] {segment.text.strip()}", end="", flush=True)

                        if start_offset != crop_range[0]:
                            void_count = 0
                            start_offset = crop_range[0]
                            ChatHistory.add_messages(speaker['name'], segment.text.strip())
                        else:
                            void_count += 1
                            if void_count > 10:
                                print("\nINFO: Getting conversation recommendation...")
                                get_recommendation()
                                #Thread(target=get_recommendation).start()

                        del portion, speaker
                        torch.cuda.empty_cache()
                    except:
                        error_count += 1
                        continue

                gc.collect()
        except KeyboardInterrupt:
            print("Recording stopped by user")
        finally:
            print("")
            RECORD_PARAMS['interrupted'] = True
            record_thread.join()
            print("\n-------------------------------------------")
            print("INFO: Conversation ended. Error count:", error_count)
            print("-------------------------------------------")

    recoder.terminate()
