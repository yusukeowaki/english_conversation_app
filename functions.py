# functions.py

import os
import time
import hashlib
import wave
import pyaudio
import streamlit as st
from pydub import AudioSegment, silence
from audiorecorder import audiorecorder
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.chains import ConversationChain
import constants as ct


# =========================================================
# 必要ディレクトリを用意
# =========================================================
os.makedirs(ct.AUDIO_INPUT_DIR, exist_ok=True)
os.makedirs(ct.AUDIO_OUTPUT_DIR, exist_ok=True)


# =========================================================
# Whisper キャッシュ
# =========================================================
@st.cache_data(show_spinner=False)
def cached_transcribe(file_bytes: bytes):
    """Whisper の文字起こし結果をファイルキャッシュ"""
    key = hashlib.md5(file_bytes).hexdigest()
    cache_path = f"{ct.AUDIO_INPUT_DIR}/cache_{key}.txt"

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    with open("tmp.wav", "wb") as f:
        f.write(file_bytes)

    try:
        with open("tmp.wav", "rb") as f:
            transcript = st.session_state.openai_obj.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
            )
        text = transcript.text
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        st.error(f"Whisper処理でエラー: {e}")
        return ""
    finally:
        if os.path.exists("tmp.wav"):
            try:
                os.remove("tmp.wav")
            except:
                pass


# =========================================================
# 録音（無音トリム込み）
# =========================================================
def record_audio(audio_input_file_path: str):
    """音声入力 → 無音カット → wav 保存"""
    audio = audiorecorder(
        start_prompt="発話開始",
        pause_prompt="やり直す",
        stop_prompt="発話終了",
        start_style={"color": "white", "background-color": "black"},
        pause_style={"color": "gray", "background-color": "white"},
        stop_style={"color": "white", "background-color": "black"},
    )

    if len(audio) == 0:
        st.warning("音声が検出されませんでした。もう一度お試しください。")
        st.stop()

    audio.export(audio_input_file_path, format="wav")

    # 無音トリム（失敗しても続行）
    try:
        sound = AudioSegment.from_wav(audio_input_file_path)
        non_silent = silence.detect_nonsilent(
            sound, silence_thresh=sound.dBFS - 16, min_silence_len=500
        )
        if non_silent:
            start_trim, end_trim = non_silent[0][0], non_silent[-1][1]
            trimmed = sound[start_trim:end_trim]
            trimmed.export(audio_input_file_path, format="wav")
    except Exception as e:
        st.warning(f"無音カット中に問題が発生: {e}")


# =========================================================
# 文字起こし
# =========================================================
def transcribe_audio(audio_input_file_path: str):
    """音声ファイル → Whisper でテキスト変換"""
    try:
        with open(audio_input_file_path, "rb") as f:
            file_bytes = f.read()
        text = cached_transcribe(file_bytes)

        # 入力ファイルは使い終わったら削除
        try:
            os.remove(audio_input_file_path)
        except:
            pass

        class Result:
            def __init__(self, text):
                self.text = text

        return Result(text)
    except Exception as e:
        st.error(f"文字起こしでエラー: {e}")

        class Empty:
            text = ""

        return Empty()


# =========================================================
# 音声ファイル保存（mp3 → wav）
# =========================================================
def save_to_wav(llm_response_audio: bytes, audio_output_file_path: str):
    """mp3 バイト列を一時保存 → wav に変換して保存"""
    tmp_mp3 = f"{ct.AUDIO_OUTPUT_DIR}/temp_{int(time.time())}.mp3"
    try:
        with open(tmp_mp3, "wb") as f:
            f.write(llm_response_audio)
        audio_mp3 = AudioSegment.from_file(tmp_mp3, format="mp3")
        audio_mp3.export(audio_output_file_path, format="wav")
    finally:
        if os.path.exists(tmp_mp3):
            try:
                os.remove(tmp_mp3)
            except:
                pass


# =========================================================
# 再生（Cloud/ローカル自動切替）
# =========================================================
def play_wav(audio_output_file_path: str, speed: float = 1.0, keep_file: bool = False):
    """
    - Streamlit Cloud：st.audio でクライアント再生（PyAudioは使用しない）
    - ローカル：環境変数 USE_PYAUDIO=1 のとき PyAudio で再生
    - speed != 1.0 のときはフレームレート変更で速度調整
    - keep_file=True のときは再生後にファイル削除しない
    """
    use_pyaudio = os.environ.get("USE_PYAUDIO", "0") == "1"

    # 速度調整（ファイルを書き換える）
    try:
        audio = AudioSegment.from_wav(audio_output_file_path)
        if speed != 1.0:
            modified = audio._spawn(
                audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed)}
            )
            modified = modified.set_frame_rate(audio.frame_rate)
            modified.export(audio_output_file_path, format="wav")
    except Exception as e:
        st.error(f"再生用の前処理でエラー: {e}")

    # ---- Cloud：st.audio 再生（推奨）----
    if not use_pyaudio:
        try:
            with open(audio_output_file_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
        except Exception as e:
            st.error(f"st.audio 再生でエラー: {e}")
        finally:
            if not keep_file and os.path.exists(audio_output_file_path):
                try:
                    os.remove(audio_output_file_path)
                except:
                    pass
        return

    # ---- ローカル：PyAudio 再生（Cloudでは不可）----
    wf = p = stream = None
    try:
        wf = wave.open(audio_output_file_path, "rb")
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
    except Exception as e:
        st.error(f"PyAudio 再生でエラー: {e}")
    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            if p is not None:
                p.terminate()
            if wf is not None:
                wf.close()
        except:
            pass
        if not keep_file and os.path.exists(audio_output_file_path):
            try:
                os.remove(audio_output_file_path)
            except:
                pass


# =========================================================
# LLM チェーン作成
# =========================================================
def create_chain(system_template: str):
    """LangChain の ConversationChain を作成"""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    return ConversationChain(
        llm=st.session_state.llm,
        memory=st.session_state.memory,
        prompt=prompt,
    )


# =========================================================
# 日本語レベル → CEFR 変換
# =========================================================
def convert_level_to_cefr(level_jp: str) -> str:
    """日本語レベル表記を CEFR (A2/B1/B2) に変換"""
    mapping = {"初級者": "A2", "中級者": "B1", "上級者": "B2"}
    return mapping.get(level_jp, "B1")


# =========================================================
# 問題生成 + 読み上げ（互換用）
# =========================================================
def create_problem_and_play_audio():
    """
    ユーザーのレベルに応じた英文を1つ生成し、TTSで1回再生して返す。
    既存コード互換用。main.py 側では文生成とTTSを分けてもOK。
    """
    user_level = st.session_state.get("englv", "中級者")
    cefr_level = convert_level_to_cefr(user_level)

    # プロンプトを整形
    system_template = ct.SYSTEM_TEMPLATE_CREATE_PROBLEM.format(level=cefr_level)
    chain = create_chain(system_template)

    # 英文生成
    problem = chain.predict(input="Generate one English sentence for practice.")
    problem = problem.strip().replace('"', "")

    # TTS音声生成と再生
    try:
        llm_response_audio = st.session_state.openai_obj.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=problem,
        )
        audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
        save_to_wav(llm_response_audio.content, audio_output_file_path)
        play_wav(audio_output_file_path, st.session_state.speed)  # 1回再生（自動削除）
        return problem, llm_response_audio
    except Exception as e:
        st.error(f"TTS音声生成エラー: {e}")
        return problem, None


# =========================================================
# 評価生成
# =========================================================
def create_evaluation():
    """ディクテーション／シャドーイング共通の評価生成"""
    try:
        return st.session_state.chain_evaluation.predict(input="")
    except Exception as e:
        st.error(f"評価生成エラー: {e}")
        return "評価を生成できませんでした。"
