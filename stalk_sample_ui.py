from multiprocessing import Process, Manager
from time import sleep
import functools
import asyncio
import sys
import os
import gc

from stalk_streamer import recoder, record_audio, CACHE_FOLDER, OUTPUT_FILENAME
import flet as ft
from playsound import playsound


USE_SPEAKER_PRESET = True


if __name__ == '__main__':
    import torch
    import torchaudio
    from pyannote.core import Segment


FONT_DIR = os.path.join(".", "res", "font")
ICON_DIR = os.path.join(".", "res", "image")
SOUND_DIR = os.path.join(os.getcwd(), "res", "sound")

ICONS = {
    'chat': os.path.join(ICON_DIR, "ic_chat.png"),
    'discover': os.path.join(ICON_DIR, "ic_discover.png"),
    'voice_recognition': os.path.join(ICON_DIR, "ic_voice_recognition.png"),
    'profile': os.path.join(ICON_DIR, "ic_profile.png"),
    'heart': os.path.join(ICON_DIR, "ic_heart.png")
}

POPUP_SOUND = os.path.join(SOUND_DIR, "pop.mp3")

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


BG_COLOR = "#F5F3FE"
FG_COLOR = "#6652AB"


def main(page: ft.Page):
    page.padding = 30
    page.bgcolor = BG_COLOR
    page.window_full_screen = True


    #
    # Load Fonts
    page.fonts = dict()
    for file in os.listdir(FONT_DIR):
        if file.endswith(".ttf"):
            print(f"INFO: Loading font: {file}")
            page.fonts[file.split(".")[0]] = os.path.join(FONT_DIR, file)

    page.theme = ft.Theme(font_family="SUITE")

    #
    # Conversation Page
    def set_name(index, name):
        target = []
        rotate_to = ft.Rotate(angle=3.14 / 2)
        if index == 0:
            target = speaker1_name.controls
        elif index == 1:
            target = speaker2_name.controls
            name = reversed(name)
            rotate_to = ft.Rotate(angle=3.14 / 2 * 3)

        for char in name:
            target.append(ft.Text(
                char, size=20, rotate=rotate_to, width=20, height=20, max_lines=1,
                overflow=ft.TextOverflow.VISIBLE, text_align=ft.TextAlign.LEFT
            ))
        page.update()

    speaker1_name = ft.Column(controls=[], spacing=0)
    speaker2_name = ft.Column(controls=[], spacing=0)

    class ShadowedCard(ft.Container):
        def __init__(self, controls, spacing=10, padding=8, radius=16, *args, **kwargs):
            content = ft.Column(
                controls=controls,
                alignment=ft.MainAxisAlignment.END,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=spacing
            )
            bgcolor = "#F1F0FA"
            shadow = ft.BoxShadow(10, 40, "#3C6652A9")
            border = ft.BorderSide(color=ft.colors.WHITE, width=1)
            border_radius = ft.BorderRadius(*[radius] * 4)
            super().__init__(
                content, *args, **kwargs, bgcolor=bgcolor, shadow=shadow,
                border_radius=border_radius, padding=padding, border=ft.Border(*[border] * 4)
            )

    header = ft.Row(
        controls=[
            ft.Text(
                "Current Conversation",
                weight=ft.FontWeight.BOLD,
                text_align=ft.TextAlign.CENTER,
                size=20,
                color="#3A393B"
            )
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    conversation_box = ft.Column(
        controls=[

        ],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.END,
        scroll=ft.ScrollMode.AUTO,
        auto_scroll=True,
        expand=True
    )

    last_conversation = None

    def add_conversation(content=""):
        nonlocal last_conversation
        last_conversation = ft.Text(content, size=18)
        conversation_box.controls.append(last_conversation)
        page.update()

    def modify_last_conversation(content):
        if last_conversation:
            last_conversation.value = content
        else:
            add_conversation(content)
        page.update()

    def add_recommendation(content=""):
        text = ft.Text(content, size=18, text_align=ft.TextAlign.CENTER)

        async def play_sound():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, functools.partial(playsound, POPUP_SOUND))

        conversation_box.controls.append(ShadowedCard(
            controls=[text], padding=18, margin=ft.Margin(10, 20, 10, 20)
        ))

        page.update()

        asyncio.run(play_sound())

        return text

    sidebar_start = ShadowedCard(
        controls=[
            ft.Container(speaker1_name, margin=ft.Margin(10, 10, 0, 10)),
            ft.Icon(name=ft.icons.BATTERY_3_BAR_OUTLINED, color="#8A75E0", size=30),
            ft.Text("40", size=16, color="#8A75E0"),
            ft.Image(src=ICONS['heart'], color="#8A75E0", width=30, height=30),
            ft.Text("31°", size=16, color="#8A75E0"),
        ],
        width=60
    )

    sidebar_end = ShadowedCard(
        controls=[
            ft.Container(speaker2_name, margin=ft.Margin(0, 10, 10, 10)),
            ft.Icon(name=ft.icons.BATTERY_6_BAR_OUTLINED, color="#C5614D", size=30),
            ft.Text("100", size=16, color="#C5614D"),
            ft.Image(src=ICONS['heart'], color="#C5614D", width=30, height=30),
            ft.Text("36°", size=16, color="#C5614D"),
        ],
        width=60
    )

    footer = ft.Row(
        controls=[
            ft.Container(
                ft.Image(src=ICONS['profile'], width=50, height=50),
                border_radius=ft.BorderRadius(*[12] * 4)),
            ft.Column(controls=[
                ft.Text("기프랩 IceBreaking", size=18, weight=ft.FontWeight.BOLD, color=ft.colors.BLACK),
                ft.Text("last conversation yesterday", size=16, color=ft.colors.GREY)
            ], spacing=0, expand=True),
            ft.Container(
                ft.IconButton(
                    content=ft.Column(controls=[
                        ft.Image(ICONS['discover'], width=40, height=40),
                        ft.Text("Discover", size=12, color="#7971A3")
                    ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    tooltip="Go to discover page",
                    style=ft.ButtonStyle(overlay_color=ft.colors.TRANSPARENT)
                ),
                margin=ft.Margin(22, 0, 22, 0),
                height=56
            ),
            ft.Container(
                ft.IconButton(
                    content=ft.Column(controls=[
                        ft.Image(ICONS['voice_recognition'], width=56, height=56)
                    ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    tooltip="Stop voice recognition",
                    style=ft.ButtonStyle(overlay_color=ft.colors.TRANSPARENT),
                    on_click=lambda _: page.window_destroy()
                ),
                shadow=ft.BoxShadow(1, 25, "#3C6652A9", offset=(0, 20)),
                height=56
            ),
            ft.Container(
                ft.IconButton(
                    content=ft.Column(controls=[
                        ft.Image(ICONS['chat'], width=40, height=40),
                        ft.Text("Talk", size=12, color="#7971A3")
                    ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    tooltip="Go to chat page",
                    style=ft.ButtonStyle(overlay_color=ft.colors.TRANSPARENT)
                ),
                margin=ft.Margin(22, 0, 22, 0),
                height=56
            )
        ],
        vertical_alignment=ft.CrossAxisAlignment.END
    )

    page.add(
        header,
        ft.Row(
            controls=[
                ft.Column([sidebar_start], alignment=ft.MainAxisAlignment.CENTER),
                ft.Container(conversation_box, margin=ft.Margin(30, 10, 30, 10), expand=True),
                ft.Column([sidebar_end], alignment=ft.MainAxisAlignment.CENTER),
            ],
            alignment=ft.MainAxisAlignment.SPACE_AROUND,
            expand=True,
        ),
        footer
    )

    #
    # Show Greetings
    message_finished = False

    def close_greetings(_):
        if message_finished:
            page.dialog.open = False
            page.update()

    greetings_message = ft.Text("로딩 중입니다. 잠시만 기다려주세요!", size=16)
    greetings = ft.AlertDialog(
        title=ft.Text("STalk AI System"),
        content=greetings_message,
        actions=[
            ft.TextButton(
                "대화 시작",
                on_click=close_greetings,
                style=ft.ButtonStyle(bgcolor=FG_COLOR, color=BG_COLOR)
            )
        ],
        bgcolor=BG_COLOR
    )
    page.dialog = greetings
    greetings.open = True
    page.update()

    #
    # Load Models
    from stalk_models import ChatHistory, llama3, whisper, audio, resnet152, token_stream

    print("\n\nInitialization finished! Press Enter to start conversation...\n")

    def get_recommendation(system_prompt, user_prompt, widget=None, init=False, auto_scroll=False):
        for chunk in llama3(system_prompt, user_prompt):
            stream = token_stream(chunk)

            if init:
                try:
                    widget = widget()
                except Exception as e:
                    print("ERROR: Failed to initialize recommendation widget", file=sys.stderr)
                    print(e, file=sys.stderr)
                init = False

            if widget:
                widget.value += stream
                if auto_scroll:
                    conversation_box.scroll_to(offset=conversation_box.height)
                page.update()

            print(stream, end="", flush=True)
        print()

    #
    # Select Speakers
    if USE_SPEAKER_PRESET:
        speaker1 = "김민서", "./SpeakerDiarization/sample_conversation/real/sentence_F.wav"
        speaker2 = "조연우", "./SpeakerDiarization/sample_conversation/real/sentence_M.wav"
        resnet152.register(*speaker1)
        set_name(0, speaker1[0])
        resnet152.register(*speaker2)
        set_name(1, speaker2[0])
    else:
        def on_speaker_selected(e: ft.FilePickerResultEvent):
            print("INFO: Selected speakers:", e.files)
            for index, file in enumerate(e.files):
                name = file.name.split(".")[0]
                resnet152.register(name, file.path)
                print(f"INFO: Speaker <{name}> registered")
                set_name(index, name)

        speaker_selector = ft.FilePicker(on_result=on_speaker_selected)
        page.overlay.append(speaker_selector)
        page.update()

        speaker_selector.pick_files(allow_multiple=True)

    #
    # Show Start Message
    greetings_message.value = ""
    get_recommendation(system_prompt, "안녕! 너는 누구야?", greetings_message)
    message_finished = True

    while greetings.open:
        sleep(0.01)

    #
    # Loop
    record_params = manager.dict(interrupted=False, duration=0.0)

    record_thread = Process(target=record_audio, kwargs=dict(params=record_params))
    record_thread.start()

    def on_window_close(e=None):
        print("")
        record_params['interrupted'] = True
        record_thread.join()
        print("\n-------------------------------------------")
        print("INFO: Conversation ended. Error count:", error_count)
        print("-------------------------------------------")

        recoder.terminate()

        print("\n\nINFO: Conversation History")
        for message in ChatHistory.messages:
            print(f"[{message['role']}] {message['content']}")

    page.on_window_close = on_window_close

    start_offset = 0.0
    temp_file = os.path.join(CACHE_FOLDER, "temp.wav")

    void_count = 0
    error_count = 0

    try:
        while not record_params['duration']:
            sleep(0.001)  # Wait until the recording starts

        while not record_params['interrupted']:  #record_thread.is_alive():
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
                        add_conversation("")
                        print()
                    else:
                        ChatHistory.remove_last()

                    modify_last_conversation(f"[{speaker['name']}] {segment.text.strip()}")
                    print(
                        f"\r({crop_range[0]:.2f}, {crop_range[1]:.2f}) in ({record_params['duration']:.2f})"
                        f" -> [{speaker['name']}] {segment.text.strip()}", end="", flush=True
                    )
                    ChatHistory.add_messages(speaker['name'], segment.text.strip())

                    if start_offset != crop_range[0]:
                        void_count = 0
                        start_offset = crop_range[0]
                    else:
                        void_count += 1
                        if void_count > 5:
                            print("\nINFO: Getting conversation recommendation...")
                            get_recommendation(
                                system_prompt, user_prompt,
                                add_recommendation, init=True, auto_scroll=True
                            )
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
        on_window_close()


if __name__ == '__main__':
    with Manager() as manager:
        ft.app(target=main, name="STalk")
