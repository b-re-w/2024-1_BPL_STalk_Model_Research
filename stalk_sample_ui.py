from multiprocessing import Process, Manager
from time import sleep
import os
import gc

from stalk_streamer import recoder, record_audio, CACHE_FOLDER, OUTPUT_FILENAME
import flet as ft


USE_SPEAKER_PRESET = True


if __name__ == '__main__':
    import torch
    import torchaudio
    from pyannote.core import Segment


FONT_DIR = os.path.join(".", "res", "font")

system_prompt = """
You are a helpful, smart, kind, and efficient Conversation Analysis and Recommendation AI System.
You always fulfill the user's requests to the best of your ability.
You need to keep listen to the conversations.
Please answer in Korean language in a bright tone.
"""

user_prompt = """
You have to recommend a topic to continue the conversation in up to 50 characters.
Please do recommend a new topic sentence related the current situation or their personal interests.
Generate text in Korean honorific and do not provide any translation description.

The format is to fill in the following <Blank>.
대화 주제: <Blank>

대화 내용 추천: <Blank>

예시 질문: <Blank>
"""


def main(page: ft.Page):
    # Load Fonts
    page.fonts = dict()
    for file in os.listdir(FONT_DIR):
        if file.endswith(".ttf"):
            print(f"INFO: Loading font: {file}")
            page.fonts[file.split(".")[0]] = os.path.join(FONT_DIR, file)

    page.theme = ft.Theme(font_family="SUITE")

    #
    # Load Models
    from stalk_models import ChatHistory, llama3, whisper, audio, resnet152, token_stream

    print("\n\nInitialization finished! Press Enter to start conversation...\n")

    def get_recommendation(system_prompt, user_prompt, widget=None):
        for chunk in llama3(system_prompt, user_prompt):
            stream = token_stream(chunk)
            if widget:
                widget.value += stream
                page.update()
            print(stream, end="", flush=True)
        print()

    #
    # Select Speakers
    if USE_SPEAKER_PRESET:
        speaker1 = "김민서", "./SpeakerDiarization/sample_conversation/real/sentence_F.wav"
        speaker2 = "조연우", "./SpeakerDiarization/sample_conversation/real/sentence_M.wav"
        resnet152.register(*speaker1)
        resnet152.register(*speaker2)
    else:
        def on_speaker_selected(e: ft.FilePickerResultEvent):
            print("INFO: Selected speakers:", e.files)
            for file in e.files:
                name = file.name.split(".")[0]
                resnet152.register(name, file.path)
                print(f"INFO: Speaker <{name}> registered")

        speaker_selector = ft.FilePicker(on_result=on_speaker_selected)
        page.overlay.append(speaker_selector)
        page.update()

        speaker_selector.pick_files(allow_multiple=True)

    #
    # Show Greetings
    message_finished = False

    def close_greetings(_):
        if message_finished:
            page.dialog.open = False
            page.update()

    greetings_message = ft.Text("")
    greetings = ft.AlertDialog(
        title=ft.Text("STalk AI System"),
        content=greetings_message,
        actions=[
            ft.TextButton(
                "대화 시작",
                on_click=close_greetings,
                style=ft.ButtonStyle(bgcolor=ft.colors.DEEP_PURPLE_500, color="white")
            )
        ],
        bgcolor=ft.colors.DEEP_PURPLE_100
    )
    page.dialog = greetings
    greetings.open = True
    page.update()

    get_recommendation(system_prompt, "안녕! 너는 누구야?", greetings_message)
    message_finished = True

    while greetings.open:
        sleep(0.01)

    #
    # Conversation Page
    def add_clicked(e):
        page.add(ft.Checkbox(label=new_task.value))
        new_task.value = ""
        page.update()

    new_task = ft.TextField(hint_text="What's needs to be done?")

    page.add(new_task, ft.FloatingActionButton(icon=ft.icons.ADD, on_click=add_clicked))

    #
    # Loop
    record_params = manager.dict(interrupted=False, duration=0.0)

    record_thread = Process(target=record_audio, kwargs=dict(params=record_params))
    record_thread.start()

    start_offset = 0.0
    temp_file = os.path.join(CACHE_FOLDER, "temp.wav")

    void_count = 0
    error_count = 0

    try:
        while not record_params['duration']:
            sleep(0.001)  # Wait until the recording starts

        while record_thread.is_alive():
            audio_range = Segment(start_offset, record_params['duration'])
            torchaudio.save(temp_file, *audio.crop(OUTPUT_FILENAME, audio_range))

            segments, info = whisper(temp_file, beam_size=5, word_timestamps=False)
            segments = iter(segments)

            for segment in segments:
                try:
                    crop_range = (start_offset + segment.start, start_offset + segment.end)
                    portion = audio.crop(OUTPUT_FILENAME, Segment(crop_range[0], crop_range[1]))
                    torchaudio.save(os.path.join(CACHE_FOLDER, f"{crop_range[0]:.2f}.wav"), *portion)

                    speaker = resnet152(*portion)

                    if start_offset != crop_range[0]:
                        print()

                    print(
                        f"\r({crop_range[0]:.f2}, {crop_range[1]:.f2} / {record_params['duration']:.f2})"
                        f" -> [{speaker['name']}] {segment.text.strip()}", end="", flush=True
                    )

                    if start_offset != crop_range[0]:
                        void_count = 0
                        start_offset = crop_range[0]
                        ChatHistory.add_messages(speaker['name'], segment.text.strip())
                    else:
                        void_count += 1
                        if void_count > 10:
                            print("\nINFO: Getting conversation recommendation...")
                            get_recommendation(system_prompt, user_prompt)
                            print("\nINFO: Finished conversation recommendation.")
                            void_count = 0

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
        record_params['interrupted'] = True
        record_thread.join()
        print("\n-------------------------------------------")
        print("INFO: Conversation ended. Error count:", error_count)
        print("-------------------------------------------")

    recoder.terminate()


if __name__ == '__main__':
    with Manager() as manager:
        ft.app(target=main, name="STalk")
