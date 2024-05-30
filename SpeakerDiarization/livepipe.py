import os
import shutil
import wave

import numpy as np
import pyaudio
import threading
#import whisper_live.utils as utils

import torch
import torchaudio

from pyannote.audio import Audio
from pyannote.core import Segment


class AudioProcessPipeline:
    """
    Pipeline for handling audio recording and streaming.

    Args:
        whisper (WhisperModel): The fast whisper model to use for transcription.
        resnet (ResNetModel): The ResNet model to use for speaker diarization.
        language (str, optional): The language code to use for transcription. Default is None.
        save_output_recording (bool, optional): Indicates whether to save recording from microphone.
        output_recording_filename (str, optional): File to save the output recording.
    """
    def __init__(self, whisper, resnet, language=None, save_output_recording=False, output_recording_filename="./output_recording.wav"):
        self.whisper = whisper
        self.resnet = resnet
        self.language = language

        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 2
        self.rate = 16000
        self.record_seconds = 60000
        self.save_output_recording = save_output_recording
        self.output_recording_filename = output_recording_filename
        self.frames = b""

        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
        except OSError as error:
            print(f"[WARN]: Unable to access microphone. {error}")
            self.stream = None

    def __call__(self, audio=None, save_file=None):
        """
        Start the transcription process.
        If an audio file is provided, it will be played and streamed to the server; otherwise, it will perform live recording.

        Args:
            audio (str, optional): Path to an audio file for transcription. Default is None, which triggers live recording.
        """

        print("[INFO]: Ready for processing!")
        #if audio is not None:
        #    resampled_file = utils.resample(audio)
        #    self.play_file(resampled_file)
        #else:
        self.record()

    def flush(self, bytes):
        """
        Do the model inference.
        """
        ndarr = np.frombuffer(bytes, np.float32)
        segments, info = self.whisper.transcribe(
            ndarr, beam_size=5, language=self.language, word_timestamps=False
        )
        for segment in segments:
            start_sample, end_sample = int(segment.start * self.rate), int(segment.end * self.rate)
            speaker = recognize(self.resnet, torch.tensor(ndarr[start_sample:end_sample]), self.rate)
            print("[%s] [%.2fs -> %.2fs] %s" % (speaker['name'], segment.start, segment.end, segment.text))

    def play_file(self, filename):
        """
        Play an audio file and send it to the server for processing.

        Reads an audio file, plays it through the audio output, and simultaneously sends
        the audio data to the server for processing. It uses PyAudio to create an audio
        stream for playback. The audio data is read from the file in chunks, converted to
        floating-point format, and sent to the server using WebSocket communication.
        This method is typically used when you want to process pre-recorded audio and send it
        to the server in real-time.

        Args:
            filename (str): The path to the audio file to be played and sent to the server.
        """
        pass

#        # read audio and create pyaudio stream
#        with wave.open(filename, "rb") as wavfile:
#            self.stream = self.p.open(
#                format=self.p.get_format_from_width(wavfile.getsampwidth()),
#                channels=wavfile.getnchannels(),
#                rate=wavfile.getframerate(),
#                input=True,
#                output=True,
#                frames_per_buffer=self.chunk,
#            )
#            try:
#                while any(client.recording for client in self.clients):
#                    data = wavfile.readframes(self.chunk)
#                    if data == b"":
#                        break
#
#                    audio_array = self.bytes_to_float_array(data)
#                    self.flush(audio_array.tobytes())
#                    self.stream.write(data)
#
#                wavfile.close()
#
#                self.flush(Client.END_OF_AUDIO.encode('utf-8'), True)
#                self.stream.close()
#
#            except KeyboardInterrupt:
#                wavfile.close()
#                self.stream.stop_stream()
#                self.stream.close()
#                self.p.terminate()
#                print("[INFO]: Keyboard interrupt.")

    def handle_ffmpeg_process(self, process, stream_type):
        print(f"[INFO]: Connecting to {stream_type} stream...")
        try:
            # Process the stream
            while True:
                in_bytes = process.stdout.read(self.chunk * 2)  # 2 bytes per sample
                if not in_bytes:
                    break
                audio_array = self.bytes_to_float_array(in_bytes)
                self.flush(audio_array.tobytes())

        except Exception as e:
            print(f"[ERROR]: Failed to connect to {stream_type} stream: {e}")
        finally:
            if process:
                process.kill()

        print(f"[INFO]: {stream_type} stream processing finished.")

    def save_chunk(self, n_audio_file):
        """
        Saves the current audio frames to a WAV file in a separate thread.

        Args:
        n_audio_file (int): The index of the audio file which determines the filename.
                            This helps in maintaining the order and uniqueness of each chunk.
        """
        t = threading.Thread(
            target=self.write_audio_frames_to_file,
            args=(self.frames[:], f"chunks/{n_audio_file}.wav",),
        )
        t.start()

    def finalize_recording(self, n_audio_file):
        """
        Finalizes the recording process by saving any remaining audio frames,
        closing the audio stream, and terminating the process.

        Args:
        n_audio_file (int): The file index to be used if there are remaining audio frames to be saved.
                            This index is incremented before use if the last chunk is saved.
        """
        if self.save_output_recording and len(self.frames):
            self.write_audio_frames_to_file(
                self.frames[:], f"chunks/{n_audio_file}.wav"
            )
            n_audio_file += 1
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        if self.save_output_recording:
            self.write_output_recording(n_audio_file)

    def record(self):
        """
        Record audio data from the input stream and save it to a WAV file.

        Continuously records audio data from the input stream, sends it to the server via a WebSocket
        connection, and simultaneously saves it to multiple WAV files in chunks. It stops recording when
        the `RECORD_SECONDS` duration is reached or when the `RECORDING` flag is set to `False`.

        Audio data is saved in chunks to the "chunks" directory. Each chunk is saved as a separate WAV file.
        The recording will continue until the specified duration is reached or until the `RECORDING` flag is set to `False`.
        The recording process can be interrupted by sending a KeyboardInterrupt (e.g., pressing Ctrl+C). After recording,
        the method combines all the saved audio chunks into the specified `out_file`.
        """
        n_audio_file = 0
        if self.save_output_recording:
            if os.path.exists("chunks"):
                shutil.rmtree("chunks")
            os.makedirs("chunks")
        try:
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames += data

                audio_array = self.bytes_to_float_array(data)

                self.flush(audio_array.tobytes())

                # save frames if more than a minute
                if len(self.frames) > 60 * self.rate:
                    if self.save_output_recording:
                        self.save_chunk(n_audio_file)
                        n_audio_file += 1
                    self.frames = b""

        except KeyboardInterrupt:
            self.finalize_recording(n_audio_file)

    def write_audio_frames_to_file(self, frames, file_name):
        """
        Write audio frames to a WAV file.

        The WAV file is created or overwritten with the specified name. The audio frames should be
        in the correct format and match the specified channel, sample width, and sample rate.

        Args:
            frames (bytes): The audio frames to be written to the file.
            file_name (str): The name of the WAV file to which the frames will be written.

        """
        with wave.open(file_name, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            wavfile.writeframes(frames)

    def write_output_recording(self, n_audio_file):
        """
        Combine and save recorded audio chunks into a single WAV file.

        The individual audio chunk files are expected to be located in the "chunks" directory. Reads each chunk
        file, appends its audio data to the final recording, and then deletes the chunk file. After combining
        and saving, the final recording is stored in the specified `out_file`.

        Args:
            n_audio_file (int): The number of audio chunk files to combine.
            out_file (str): The name of the output WAV file to save the final recording.

        """
        input_files = [
            f"chunks/{i}.wav"
            for i in range(n_audio_file)
            if os.path.exists(f"chunks/{i}.wav")
        ]
        with wave.open(self.output_recording_filename, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            for in_file in input_files:
                with wave.open(in_file, "rb") as wav_in:
                    while True:
                        data = wav_in.readframes(self.chunk)
                        if data == b"":
                            break
                        wavfile.writeframes(data)
                # remove this file
                os.remove(in_file)
        wavfile.close()
        # clean up temporary directory to store chunks
        if os.path.exists("chunks"):
            shutil.rmtree("chunks")

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        """
        Convert audio data from bytes to a NumPy float array.

        It assumes that the audio data is in 16-bit PCM format. The audio data is normalized to
        have values between -1 and 1.

        Args:
            audio_bytes (bytes): Audio data in bytes.

        Returns:
            np.ndarray: A NumPy array containing the audio data as float values normalized between -1 and 1.
        """
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0


def extract_embedding(resnet, pcm, sample_rate):
    pcm = pcm.to(torch.float)
    if sample_rate != resnet.resample_rate:
        pcm = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resnet.resample_rate)(pcm)
    feats = resnet.compute_fbank(
        pcm,
        sample_rate=resnet.resample_rate,
        cmn=True
    )
    feats = feats.unsqueeze(0)
    feats = feats.to(resnet.device)
    resnet.model.eval()
    with torch.no_grad():
        outputs = resnet.model(feats)
        outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
    embedding = outputs[0].to(torch.device('cpu'))
    return embedding


def recognize(resnet, pcm, sample_rate):
    q = extract_embedding(resnet, pcm, sample_rate)
    best_score = 0.0
    best_name = ''
    for name, e in resnet.table.items():
        score = resnet.cosine_similarity(q, e)
        if best_score < score:
            best_score = score
            best_name = name
    result = {'name': best_name, 'confidence': best_score}
    return result


if __name__ == "__main__":
    save_output_recording = False
    output_recording_filename = "./output_recording.wav"

    if save_output_recording and not output_recording_filename.endswith(".wav"):
        raise ValueError(f"Please provide a valid `output_recording_filename`: {output_recording_filename}")

    AudioProcessPipeline(
        whisper=None,
        resnet=None,
        save_output_recording=save_output_recording,
        output_recording_filename=output_recording_filename
    )
