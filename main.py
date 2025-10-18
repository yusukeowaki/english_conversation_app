# main.py
# =========================================================
# 生成AI英会話アプリ（Streamlit Community Cloud 対応版）
#  - 音声の再生は st.audio（functions.play_wav）でブラウザ側再生
#  - 録音は streamlit-audiorecorder
#  - Whisper/TTS は OpenAI API
#  - 会話/評価は LangChain
# =========================================================

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
os.makedirs(ct.AUDIO_INPUT_DIR, exist_ok=True)
os.makedirs(ct.AUDIO_OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title=ct.APP_NAME)
st.markdown(f"## {ct.APP_NAME}")

# ==============================
# セッション初期化
# ==============================
if "messages" not in st.session_state:
    # UI / 状態
    st.session_state.messages = []
    st.session_state.start_flg = False
    st.session_state.pre_mode = ""
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
    st.session_state.dictation_chat_message = ""
    st.session_state.chat_open_flg = False  # 入力受付フェーズ制御
    st.session_state.problem = ""

    # OpenAI / LLM
    st.session_state.openai_obj = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",  # MessagesPlaceholder("history") と一致
    )

    # 「日常英会話」用チェーン
    st.session_state.chain_basic_conversation = ft.create_chain(
        ct.SYSTEM_TEMPLATE_BASIC_CONVERSATION
    )

# ==============================
# 画面上部 UI
# ==============================
col1, col2, col3, col4 = st.columns([2, 2, 3, 3])

with col1:
    if st.session_state.start_flg:
        st.button("開始", use_container_width=True, type="primary", disabled=True)
    else:
        st.session_state.start_flg = st.button(
            "開始", use_container_width=True, type="primary"
        )

with col2:
    st.session_state.speed = st.selectbox(
        "再生速度", options=ct.PLAY_SPEED_OPTION, index=3, label_visibility="collapsed"
    )

with col3:
    st.session_state.mode = st.selectbox(
        "モード", options=[ct.MODE_1, ct.MODE_2, ct.MODE_3], label_visibility="collapsed"
    )

    # モード切替時の初期化
    if st.session_state.mode != st.session_state.pre_mode:
        st.session_state.start_flg = False
        # 共通の整理
        st.session_state.chat_open_flg = False
        st.session_state.problem = ""

        # シャドーイング側
        st.session_state.shadowing_flg = False
        st.session_state.shadowing_in_progress = False
        st.session_state.shadowing_count = 0

        # ディクテーション側
        st.session_state.dictation_flg = False
        st.session_state.dictation_count = 0
        st.session_state.dictation_chat_message = ""

    st.session_state.pre_mode = st.session_state.mode

with col4:
    st.session_state.englv = st.selectbox(
        "英語レベル", options=ct.ENGLISH_LEVEL_OPTION, label_visibility="collapsed"
    )

with st.chat_message("assistant", avatar="images/ai_icon.jpg"):
    st.markdown(
        "こちらは生成AIによる音声英会話の練習アプリです。何度も繰り返し練習し、英語力をアップさせましょう。"
    )
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
# 既存メッセージの描画
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

# 実行ボタン（モード別）
if st.session_state.shadowing_flg:
    st.session_state.shadowing_button_flg = st.button("シャドーイング開始")
if st.session_state.dictation_flg:
    st.session_state.dictation_button_flg = st.button("ディクテーション開始")

# ディクテーション入力ガイド
if st.session_state.chat_open_flg:
    st.info("AIが読み上げた音声を、画面下部のチャット欄からそのまま入力・送信してください。")

st.session_state.dictation_chat_message = st.chat_input("※「ディクテーション」選択時以外は送信不可")
if st.session_state.dictation_chat_message and not st.session_state.chat_open_flg:
    # 入力欄が開いていないのに送信されたら無視
    st.stop()

# ==============================
# 「開始」押下後の処理本体
# ==============================
if st.session_state.start_flg:

    # --------------------------- モード：ディクテーション ---------------------------
    if st.session_state.mode == ct.MODE_3 and (
        st.session_state.dictation_button_flg
        or st.session_state.dictation_count == 0
        or st.session_state.dictation_chat_message  # 回答送信時
    ):
        # 初回のみチェーン作成（問題生成用）
        if st.session_state.dictation_first_flg:
            st.session_state.chain_create_problem = ft.create_chain(
                ct.SYSTEM_TEMPLATE_CREATE_PROBLEM
            )
            st.session_state.dictation_first_flg = False

        # まだ入力受付フェーズに入っていない → 問題提示（TTS 再生）
        if not st.session_state.chat_open_flg:
            with st.spinner("問題文生成中..."):
                st.session_state.problem, _ = ft.create_problem_and_play_audio()
            st.session_state.chat_open_flg = True   # ここで入力解禁
            st.session_state.dictation_flg = False  # ボタンの表示オフ
            st.rerun()

        # 回答が送られて来たら評価
        else:
            if not st.session_state.dictation_chat_message:
                st.stop()

            # 表示
            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(st.session_state.problem)
            with st.chat_message("user", avatar=ct.USER_ICON_PATH):
                st.markdown(st.session_state.dictation_chat_message)

            # 履歴
            st.session_state.messages.append(
                {"role": "assistant", "content": st.session_state.problem}
            )
            st.session_state.messages.append(
                {"role": "user", "content": st.session_state.dictation_chat_message}
            )

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
            st.session_state.messages.append(
                {"role": "assistant", "content": llm_response_evaluation}
            )
            st.session_state.messages.append({"role": "other"})

            # 状態更新
            st.session_state.dictation_flg = True
            st.session_state.dictation_chat_message = ""
            st.session_state.dictation_count += 1
            st.session_state.chat_open_flg = False  # 次回の問題出題へ
            st.rerun()

    # --------------------------- モード：日常英会話 ---------------------------
    if st.session_state.mode == ct.MODE_1:
        # 録音 → Whisper
        audio_input_file_path = f"{ct.AUDIO_INPUT_DIR}/audio_input_{int(time.time())}.wav"
        ft.record_audio(audio_input_file_path)

        with st.spinner("音声入力をテキストに変換中..."):
            transcript = ft.transcribe_audio(audio_input_file_path)
            audio_input_text = transcript.text

        # 入力テキストを表示
        with st.chat_message("user", avatar=ct.USER_ICON_PATH):
            st.markdown(audio_input_text)

        with st.spinner("回答の音声読み上げ準備中..."):
            # 応答生成
            llm_response = st.session_state.chain_basic_conversation.predict(
                input=audio_input_text
            )
            # TTS → wav 保存
            tts_resp = st.session_state.openai_obj.audio.speech.create(
                model="tts-1", voice="alloy", input=llm_response
            )
            out_wav = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
            ft.save_to_wav(tts_resp, out_wav)

        # 再生
        ft.play_wav(out_wav, speed=st.session_state.speed)

        # 応答表示 & 履歴
        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(llm_response)
        st.session_state.messages.append({"role": "user", "content": audio_input_text})
        st.session_state.messages.append({"role": "assistant", "content": llm_response})

    # --------------------------- モード：シャドーイング ---------------------------
    if st.session_state.mode == ct.MODE_2 and (
        st.session_state.shadowing_button_flg
        or st.session_state.shadowing_count == 0
        or st.session_state.shadowing_in_progress
    ):
        # 初回だけ生成チェーン
        if st.session_state.shadowing_first_flg:
            st.session_state.chain_create_problem = ft.create_chain(
                ct.SYSTEM_TEMPLATE_CREATE_PROBLEM
            )
            st.session_state.shadowing_first_flg = False

        # 進行中でなければ「問題決定 → TTS 2回再生 → 録音へ」
        if not st.session_state.shadowing_in_progress:
            custom_sentence = st.text_input(
                "読み上げたい英文を入力（空欄ならAIが自動生成します）",
                placeholder="例：It's a beautiful day today, isn't it?",
            )

            # 表示ON/OFF
            if "show_text_flg" not in st.session_state:
                st.session_state.show_text_flg = True
            st.session_state.show_text_flg = st.checkbox(
                "英文を表示する（ONで見ながら練習）",
                value=st.session_state.show_text_flg,
                key="show_text_checkbox",
            )

            # 問題文決定
            if custom_sentence:
                st.session_state.problem = custom_sentence
            else:
                with st.spinner("問題文生成中..."):
                    st.session_state.problem, _ = ft.create_problem_and_play_audio()

            # 今回の評価対象を固定
            st.session_state.shadowing_eval_target = st.session_state.problem

            # 表示（履歴重複を避ける）
            if st.session_state.show_text_flg:
                with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                    st.markdown(f"**練習文:** {st.session_state.problem}")
            if not st.session_state.messages or st.session_state.messages[-1].get(
                "content"
            ) != f"**練習文:** {st.session_state.problem}":
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"**練習文:** {st.session_state.problem}"}
                )

            # TTS 2回再生（同一ファイルを使い回し）
            tts_resp = st.session_state.openai_obj.audio.speech.create(
                model="tts-1", voice="alloy", input=st.session_state.problem
            )
            tmp_wav = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
            ft.save_to_wav(tts_resp, tmp_wav)

            st.info("🔊【1回目】聞き取り練習中...")
            ft.play_wav(tmp_wav, st.session_state.speed, keep_file=True)
            st.info("🗣️【2回目】シャドーイング練習スタート！")
            ft.play_wav(tmp_wav, st.session_state.speed, keep_file=False)

            # 録音フェーズへ移行
            st.session_state.shadowing_in_progress = True

        # ここから録音 → 文字起こし → 評価
        if st.session_state.shadowing_in_progress:
            # 録音
            audio_in = f"{ct.AUDIO_INPUT_DIR}/audio_input_{int(time.time())}.wav"
            ft.record_audio(audio_in)

            # 文字起こし
            with st.spinner("音声入力をテキストに変換中..."):
                transcript = ft.transcribe_audio(audio_in)
                user_text = transcript.text

            if not user_text.strip():
                st.warning("音声が検出されませんでした。もう一度お試しください。")
                st.stop()

            # 表示
            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(st.session_state.shadowing_eval_target)
            with st.chat_message("user", avatar=ct.USER_ICON_PATH):
                st.markdown(user_text)

            # 履歴
            st.session_state.messages.append({"role": "user", "content": user_text})

            # 評価
            with st.spinner("評価結果の生成中..."):
                system_template = ct.SYSTEM_TEMPLATE_EVALUATION.format(
                    llm_text=st.session_state.shadowing_eval_target, user_text=user_text
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
