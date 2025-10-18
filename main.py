# main.py
import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

import functions as ft
import constants as ct


# ==============================
# 初期設定
# ==============================
load_dotenv()

# 必要ディレクトリ（保険）
os.makedirs(ct.AUDIO_INPUT_DIR, exist_ok=True)
os.makedirs(ct.AUDIO_OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title=ct.APP_NAME)
st.markdown(f"## {ct.APP_NAME}")

# ==============================
# セッション初期化
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

    # 共通フラグ
    st.session_state.start_flg = False
    st.session_state.pre_mode = ""
    st.session_state.problem = ""
    st.session_state.speed = 1.0

    # シャドーイング
    st.session_state.shadowing_flg = False
    st.session_state.shadowing_button_flg = False
    st.session_state.shadowing_count = 0
    st.session_state.shadowing_first_flg = True
    st.session_state.shadowing_in_progress = False
    st.session_state.shadowing_eval_target = ""

    # ディクテーション
    st.session_state.dictation_flg = False
    st.session_state.dictation_button_flg = False
    st.session_state.dictation_count = 0
    st.session_state.dictation_first_flg = True
    st.session_state.chat_open_flg = False
    st.session_state.dictation_chat_message = ""
    st.session_state.dictation_tts_bytes = None  # ← 必ずWAVの生バイトを入れる

    # OpenAI & LangChain
    st.session_state.openai_obj = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",
    )

    # 「日常英会話」チェーン
    st.session_state.chain_basic_conversation = ft.create_chain(
        ct.SYSTEM_TEMPLATE_BASIC_CONVERSATION
    )

# ==============================
# 画面上部 UI
# ==============================
col1, col2, col3, col4 = st.columns([2, 2, 3, 3])

with col1:
    if st.session_state.start_flg:
        st.button("開始", use_container_width=True, type="primary")
    else:
        st.session_state.start_flg = st.button("開始", use_container_width=True, type="primary")

with col2:
    st.session_state.speed = st.selectbox(
        label="再生速度",
        options=ct.PLAY_SPEED_OPTION,
        index=3,
        label_visibility="collapsed",
    )

with col3:
    st.session_state.mode = st.selectbox(
        label="モード",
        options=[ct.MODE_1, ct.MODE_2, ct.MODE_3],
        label_visibility="collapsed",
    )
    # モード変更時の初期化
    if st.session_state.mode != st.session_state.pre_mode:
        st.session_state.start_flg = False

        # ディクテの状態をリセット
        st.session_state.dictation_flg = False
        st.session_state.dictation_count = 0
        st.session_state.chat_open_flg = False
        st.session_state.dictation_chat_message = ""
        st.session_state.dictation_tts_bytes = None

        # シャドーイングの状態をリセット
        st.session_state.shadowing_flg = False
        st.session_state.shadowing_count = 0
        st.session_state.shadowing_in_progress = False
        st.session_state.shadowing_eval_target = ""

    st.session_state.pre_mode = st.session_state.mode

with col4:
    st.session_state.englv = st.selectbox(
        label="英語レベル",
        options=ct.ENGLISH_LEVEL_OPTION,
        label_visibility="collapsed",
    )

with st.chat_message("assistant", avatar="images/ai_icon.jpg"):
    st.markdown("こちらは生成AIによる音声英会話の練習アプリです。何度も繰り返し練習し、英語力をアップさせましょう。")
    st.markdown("**【操作説明】**")
    st.success(
        """
- モードと再生速度を選択し、「英会話開始」ボタンを押して英会話を始めましょう。
- モードは「日常英会話」「シャドーイング」「ディクテーション」から選べます。
- 発話後、5秒間沈黙することで音声入力が完了します。
- 「一時中断」ボタンを押すことで、英会話を一時中断できます。
"""
    )
st.divider()

# ==============================
# メッセージ履歴の表示
# ==============================
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar="images/ai_icon.jpg"):
            st.markdown(message["content"])
    elif message["role"] == "user":
        with st.chat_message("user", avatar="images/user_icon.jpg"):
            st.markdown(message["content"])
    else:
        st.divider()

# 実行ボタン類
if st.session_state.shadowing_flg:
    st.session_state.shadowing_button_flg = st.button("シャドーイング開始")
if st.session_state.dictation_flg:
    st.session_state.dictation_button_flg = st.button("ディクテーション開始")

# ディクテーション時の入力ヒント
if st.session_state.chat_open_flg:
    st.info("AIが読み上げた音声を、画面下部のチャット欄からそのまま入力・送信してください。")

# チャット入力
st.session_state.dictation_chat_message = st.chat_input("※「ディクテーション」選択時以外は送信不可")
if st.session_state.dictation_chat_message and not st.session_state.chat_open_flg:
    st.stop()

# ==============================
# 「開始」ボタン押下時の処理
# ==============================
if st.session_state.start_flg:

    # ---------- ディクテーション ----------
    if st.session_state.mode == ct.MODE_3 and (
        st.session_state.dictation_button_flg
        or st.session_state.dictation_count == 0
        or st.session_state.dictation_chat_message
    ):
        if st.session_state.dictation_first_flg:
            st.session_state.chain_create_problem = ft.create_chain(ct.SYSTEM_TEMPLATE_CREATE_PROBLEM)
            st.session_state.dictation_first_flg = False

        # まだ入力を受け付けていない ⇒ 問題出題フェーズ
        if not st.session_state.chat_open_flg:
            with st.spinner("問題文生成中..."):
                # ← functions は (problem, llm_response_audio) を返す
                #    llm_response_audio.content は MP3 バイトなので、WAVに変換して保持する
                st.session_state.problem, llm_resp = ft.create_problem_and_play_audio()

                st.session_state.dictation_tts_bytes = None
                if llm_resp is not None:
                    tmp_wav = os.path.join(
                        ct.AUDIO_OUTPUT_DIR, f"dict_{int(time.time())}.wav"
                    )
                    try:
                        # MP3 -> WAV へ変換して一旦保存
                        ft.save_to_wav(llm_resp.content, tmp_wav)
                        # WAV を生バイトとして読み込み、セッションに保持
                        with open(tmp_wav, "rb") as f:
                            st.session_state.dictation_tts_bytes = f.read()
                    finally:
                        try:
                            os.remove(tmp_wav)
                        except Exception:
                            pass

            # 出題音声の再生
            if st.session_state.dictation_tts_bytes:
                st.audio(st.session_state.dictation_tts_bytes, format="audio/wav")

            st.session_state.chat_open_flg = True
            st.session_state.dictation_flg = False
            st.rerun()

        # 回答を受け取った後の評価フェーズ
        else:
            if not st.session_state.dictation_chat_message:
                st.stop()

            # 出題文と音声（必要なら再生）・ユーザ入力の表示
            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(st.session_state.problem)
                if st.session_state.dictation_tts_bytes:
                    st.audio(st.session_state.dictation_tts_bytes, format="audio/wav")

            with st.chat_message("user", avatar=ct.USER_ICON_PATH):
                st.markdown(st.session_state.dictation_chat_message)

            # 履歴
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.problem})
            st.session_state.messages.append({"role": "user", "content": st.session_state.dictation_chat_message})

            # 評価
            with st.spinner("評価結果の生成中..."):
                system_template = ct.SYSTEM_TEMPLATE_EVALUATION.format(
                    llm_text=st.session_state.problem,
                    user_text=st.session_state.dictation_chat_message,
                )
                st.session_state.chain_evaluation = ft.create_chain(system_template)
                llm_response_evaluation = ft.create_evaluation()

            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(llm_response_evaluation)

            st.session_state.messages.append({"role": "assistant", "content": llm_response_evaluation})
            st.session_state.messages.append({"role": "other"})

            # 次回の準備
            st.session_state.dictation_flg = True
            st.session_state.dictation_chat_message = ""
            st.session_state.dictation_count += 1
            st.session_state.chat_open_flg = False
            st.session_state.dictation_tts_bytes = None

            st.rerun()

    # ---------- 日常英会話 ----------
    if st.session_state.mode == ct.MODE_1:
        audio_input_file_path = os.path.join(
            ct.AUDIO_INPUT_DIR, f"audio_input_{int(time.time())}.wav"
        )
        ft.record_audio(audio_input_file_path)

        with st.spinner("音声入力をテキストに変換中..."):
            transcript = ft.transcribe_audio(audio_input_file_path)
            audio_input_text = transcript.text

        with st.chat_message("user", avatar=ct.USER_ICON_PATH):
            st.markdown(audio_input_text)

        with st.spinner("回答の音声読み上げ準備中..."):
            llm_response = st.session_state.chain_basic_conversation.predict(input=audio_input_text)
            tts = st.session_state.openai_obj.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=llm_response,
            )
            out_path = os.path.join(
                ct.AUDIO_OUTPUT_DIR, f"audio_output_{int(time.time())}.wav"
            )
            ft.save_to_wav(tts.content, out_path)

        # 再生
        ft.play_wav(out_path, speed=st.session_state.speed)

        # 表示・履歴
        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(llm_response)

        st.session_state.messages.append({"role": "user", "content": audio_input_text})
        st.session_state.messages.append({"role": "assistant", "content": llm_response})

    # ---------- シャドーイング ----------
    if st.session_state.mode == ct.MODE_2 and (
        st.session_state.shadowing_button_flg
        or st.session_state.shadowing_count == 0
        or st.session_state.get("shadowing_in_progress", False)
    ):
        if st.session_state.shadowing_first_flg:
            st.session_state.chain_create_problem = ft.create_chain(ct.SYSTEM_TEMPLATE_CREATE_PROBLEM)
            st.session_state.shadowing_first_flg = False

        # 進行中でなければ「問題決定→TTS 2回再生→録音へ」
        if not st.session_state.shadowing_in_progress:
            custom_sentence = st.text_input(
                "読み上げたい英文を入力（空欄ならAIが自動生成します）",
                placeholder="例：It's a beautiful day today, isn't it?",
            )

            if "show_text_flg" not in st.session_state:
                st.session_state.show_text_flg = True
            st.session_state.show_text_flg = st.checkbox(
                "英文を表示する（ONで見ながら練習）",
                value=st.session_state.show_text_flg,
                key="show_text_checkbox",
            )

            # 問題文の確定
            if custom_sentence:
                st.session_state.problem = custom_sentence
            else:
                with st.spinner("問題文生成中..."):
                    p, llm_resp = ft.create_problem_and_play_audio()
                    st.session_state.problem = p
                    # 1回目の再生は functions 側で実行済み

            # 表示
            if st.session_state.show_text_flg:
                with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                    st.markdown(f"**練習文:** {st.session_state.problem}")
            if not st.session_state.messages or st.session_state.messages[-1].get("content") != f"**練習文:** {st.session_state.problem}":
                st.session_state.messages.append({"role": "assistant", "content": f"**練習文:** {st.session_state.problem}"})

            # 2回目も読み上げ（ttsを新規作成）
            tts2 = st.session_state.openai_obj.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=st.session_state.problem,
            )
            out_path2 = os.path.join(
                ct.AUDIO_OUTPUT_DIR, f"audio_output_{int(time.time())}.wav"
            )
            ft.save_to_wav(tts2.content, out_path2)
            st.info("🗣️【2回目】シャドーイング練習スタート！")
            ft.play_wav(out_path2, st.session_state.speed, keep_file=False)

            # 録音フェーズへ
            st.session_state.shadowing_eval_target = st.session_state.problem
            st.session_state.shadowing_in_progress = True

        # 録音 → 文字起こし → 評価
        if st.session_state.shadowing_in_progress:
            input_path = os.path.join(
                ct.AUDIO_INPUT_DIR, f"audio_input_{int(time.time())}.wav"
            )
            ft.record_audio(input_path)

            with st.spinner("音声入力をテキストに変換中..."):
                transcript = ft.transcribe_audio(input_path)
                user_text = transcript.text

            if not user_text.strip():
                st.warning("音声が検出されませんでした。もう一度お試しください。")
                st.stop()

            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(st.session_state.shadowing_eval_target)
            with st.chat_message("user", avatar=ct.USER_ICON_PATH):
                st.markdown(user_text)

            st.session_state.messages.append({"role": "user", "content": user_text})

            with st.spinner("評価結果の生成中..."):
                system_template = ct.SYSTEM_TEMPLATE_EVALUATION.format(
                    llm_text=st.session_state.shadowing_eval_target,
                    user_text=user_text,
                )
                st.session_state.chain_evaluation = ft.create_chain(system_template)
                eval_text = ft.create_evaluation()

            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(eval_text)

            st.session_state.messages.append({"role": "assistant", "content": eval_text})
            st.session_state.messages.append({"role": "other"})

            # 後片付け
            st.session_state.shadowing_eval_target = ""
            st.session_state.shadowing_in_progress = False
            st.session_state.shadowing_flg = True
            st.session_state.shadowing_count += 1

            st.rerun()
