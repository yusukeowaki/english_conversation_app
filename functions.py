# functions.py
# ------------------------------------------------------------
# 共通関数群（Streamlit Cloud対応版：PyAudioは使わず、再生は st.audio）
# ------------------------------------------------------------

from __future__ import annotations

import io
import os
import time
import hashlib

import streamlit as st
from pydub import AudioSegment, silence
import imageio_ffmpeg  # FFmpeg バイナリを同梱的に提供
from audiorecorder import audiorecorder

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.chains import ConversationChain

import constants as ct


# pydub が使う ffmpeg の実体を imageio-ffmpeg に向ける
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# 必要ディレクトリ（Gitには空で入らない可能性があるため保険）
os.makedirs(ct.AUDIO_INPUT_DIR, exist_ok=True)
os.makedirs(ct.AUDIO_OUTPUT_DIR, exist_ok=True)


# =========================================================
# Whisper キャッシュ（同一音声は再計算しない）
# =========================================================
@st.cache_data(show_spinner=False)
def cached_transcribe(file_bytes: bytes) -> str:
    """Whisperの文字起こし結果をバイト列ハッシュでキャッシュ保存"""
    key = hashlib.md5(file_bytes).hexdigest()
    cache_path = os.path.join(ct.AUDIO_INPUT_DIR, f"cache_{key}.txt")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    tmp = "tmp_whisper.wav"
    with open(tmp, "wb") as f:
        f.write(file_bytes)

    try:
        with open(tmp, "rb") as f:
            transcript = st.session_state.openai_obj.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
            )
        text = (transcript.text or "").strip()
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        st.error(f"Whisper処理でエラー: {e}")
        return ""
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# =========================================================
# 録音（ブラウザ）→ 無音トリム → WAV保存
# =========================================================
def record_audio(audio_input_file_path: str) -> None:
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

    # 無音区間をトリム（失敗しても致命ではない）
    try:
        snd = AudioSegment.from_wav(audio_input_file_path)
        non_silent = silence.detect_nonsilent(
            snd, silence_thresh=snd.dBFS - 16, min_silence_len=500
        )
        if non_silent:
            start_trim, end_trim = non_silent[0][0], non_silent[-1][1]
            trimmed = snd[start_trim:end_trim]
            trimmed.export(audio_input_file_path, format="wav")
    except Exception as e:
        st.warning(f"無音カット中に問題が発生: {e}")


# =========================================================
# 文字起こし（Whisper）
# =========================================================
def transcribe_audio(audio_input_file_path: str):
    """音声ファイル → Whisper文字起こし。互換の簡易Resultを返す"""
    try:
        with open(audio_input_file_path, "rb") as f:
            b = f.read()
        text = cached_transcribe(b)

        try:
            os.remove(audio_input_file_path)
        except Exception:
            pass

        class Result:
            def __init__(self, text: str):
                self.text = text

        return Result(text)
    except Exception as e:
        st.error(f"文字起こしでエラー: {e}")

        class Empty:
            text = ""

        return Empty()


# =========================================================
# mp3→wav 保存（主に日常会話・シャドーイング用）
# =========================================================
def save_to_wav(llm_response_audio: bytes, audio_output_file_path: str) -> None:
    tmp_mp3 = os.path.join(ct.AUDIO_OUTPUT_DIR, f"temp_{int(time.time())}.mp3")
    try:
        with open(tmp_mp3, "wb") as f:
            f.write(llm_response_audio)
        audio_mp3 = AudioSegment.from_file(tmp_mp3, format="mp3")
        audio_mp3.export(audio_output_file_path, format="wav")
    finally:
        try:
            if os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
        except Exception:
            pass


# =========================================================
# 再生（ブラウザ側のオーディオプレイヤー）
# =========================================================
def play_wav(audio_output_file_path: str, speed: float = 1.0, keep_file: bool = False):
    """
    Streamlit の st.audio で再生する。
    - speed != 1.0 のときはフレームレート変更で速度調整
    - keep_file=False のときは再生後にファイルを削除
    """
    try:
        audio = AudioSegment.from_wav(audio_output_file_path)

        if speed != 1.0:
            audio = audio._spawn(
                audio.raw_data,
                overrides={"frame_rate": int(audio.frame_rate * speed)},
            ).set_frame_rate(audio.frame_rate)

        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        st.audio(buf.read(), format="audio/wav")
    except Exception as e:
        st.error(f"再生中にエラー: {e}")
    finally:
        if not keep_file and os.path.exists(audio_output_file_path):
            try:
                os.remove(audio_output_file_path)
            except Exception:
                pass


# =========================================================
# LangChain の会話チェーン生成
# =========================================================
def create_chain(system_template: str) -> ConversationChain:
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
    mapping = {
        "初級者": "A2",
        "中級者": "B1",
        "上級者": "B2",
    }
    return mapping.get(level_jp, "B1")


# =========================================================
# （ディクテーション向け）問題生成 + 音声（WAVバイト列）返却
# =========================================================
def create_problem_and_play_audio():
    """
    ユーザーのレベルに応じた英文を1つ生成し、TTSした音声(WAV bytes)を返す。
    画面側で st.audio に渡して再生する（rerunで消えないようにbytesを保持）。
    戻り値: (problem: str, wav_bytes: Optional[bytes])
    """
    user_level = st.session_state.get("englv", "中級者")
    cefr_level = convert_level_to_cefr(user_level)

    # 英文生成
    system_template = ct.SYSTEM_TEMPLATE_CREATE_PROBLEM.format(level=cefr_level)
    chain = create_chain(system_template)
    problem = chain.predict(input="Generate one English sentence for practice.")
    problem = problem.strip().replace('"', "")

    try:
        # OpenAI TTS -> mp3 bytes
        tts = st.session_state.openai_obj.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=problem,
        )
        mp3_bytes = tts.content

        # mp3 bytes -> pydub(AudioSegment)
        mp3_buf = io.BytesIO(mp3_bytes)
        snd = AudioSegment.from_file(mp3_buf, format="mp3")

        # 速度調整（必要なら）
        speed = float(st.session_state.get("speed", 1.0))
        if speed != 1.0:
            snd = snd._spawn(
                snd.raw_data, overrides={"frame_rate": int(snd.frame_rate * speed)}
            ).set_frame_rate(snd.frame_rate)

        # WAV bytes にして返す
        wav_buf = io.BytesIO()
        snd.export(wav_buf, format="wav")
        wav_bytes = wav_buf.getvalue()

        return problem, wav_bytes
    except Exception as e:
        st.error(f"TTS音声生成エラー: {e}")
        return problem, None


# =========================================================
# 評価生成（ディクテ/シャドーイング共通）
# =========================================================
def create_evaluation() -> str:
    try:
        return st.session_state.chain_evaluation.predict(input="")
    except Exception as e:
        st.error(f"評価生成エラー: {e}")
        return "評価を生成できませんでした。"
