# main.py
import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

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

# 必要ディレクトリの作成（なければ作る）
os.makedirs(ct.AUDIO_INPUT_DIR, exist_ok=True)
os.makedirs(ct.AUDIO_OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title=ct.APP_NAME)
st.markdown(f"## {ct.APP_NAME}")

# ==============================
# セッション初期化
# ==============================
if "initialized" not in st.session_state:
    st.session_state.initialized = True

    # UI / 状態
    st.session_state.messages = []
    st.session_state.start_flg = False
    st.session_state.pre_mode = ""
    st.session_state.speed = 1.0
    st.session_state.intro_shown = False

    # シャドーイング
    st.session_state.shadowing_flg = False
    st.session_state.shadowing_button_flg = False
    st.session_state.shadowing_count = 0
    st.session_state.shadowing_first_flg = True
    st.session_state.shadowing_audio_input_flg = False
    st.session_state.shadowing_evaluation_first_flg = True
    st.session_state.shadowing_in_progress = False
    st.session_state.shadowing_eval_target = ""

    # ディクテーション
    st.session_state.dictation_flg = False
    st.session_state.dictation_button_flg = False
    st.session_state.dictation_count = 0
    st.session_state.dictation_first_flg = True
    st.session_state.dictation_chat_message = ""
    st.session_state.dictation_evaluation_first_flg = True

    # 共通
    st.session_state.chat_open_flg = False
    st.session_state.problem = ""
    st.session_state.englv = ct.ENGLISH_LEVEL_OPTION[1]  # 既定: 中級者

    # OpenAI API キー
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY が見つかりません。.env を確認してください。")
        st.stop()

    # OpenAI & LangChain
    st.session_state.openai_obj = OpenAI(api_key=api_key)
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",  # MessagesPlaceholder("history") と一致
    )

    # 日常英会話チェーン
    st.session_state.chain_basic_conversation = ft.create_chain(
        ct.SYSTEM_TEMPLATE_BASIC_CONVERSATION
    )

# ==============================
# 画面上部 UI
# ==============================
col1, col2, col3, col4 = st.columns([2, 2, 3, 3])

with col1:
    clicked_start = st.button("開始", use_container_width=True, type="primary")
    if clicked_start:
        st.session_state.start_flg = True

with col2:
    st.session_state.speed = st.selectbox(
        label="再生速度",
        options=ct.PLAY_SPEED_OPTION,
        index=3,
        label_visibility="collapsed"
    )

with col3:
    st.session_state.mode = st.selectbox(
        label="モード",
        options=[ct.MODE_1, ct.MODE_2, ct.MODE_3],
        label_visibility="collapsed"
    )
    # モード切替時の初期化
    if st.session_state.mode != st.session_state.pre_mode:
        st.session_state.start_flg = False
        if st.session_state.mode == ct.MODE_1:
            st.session_state.dictation_flg = False
            st.session_state.shadowing_flg = False
        st.session_state.shadowing_count = 0
        if st.session_state.mode == ct.MODE_2:
            st.session_state.dictation_flg = False
        st.session_state.dictation_count = 0
        if st.session_state.mode == ct.MODE_3:
            st.session_state.shadowing_flg = False
        st.session_state.chat_open_flg = False
        # シャドーイング進行フラグは常に落とす
        st.session_state.shadowing_in_progress = False
    st.session_state.pre_mode = st.session_state.mode

with col4:
    st.session_state.englv = st.selectbox(
        label="英語レベル",
        options=ct.ENGLISH_LEVEL_OPTION,
        index=ct.ENGLISH_LEVEL_OPTION.index(st.session_state.get("englv", "中級者")),
        label_visibility="collapsed"
    )

# 初回のみ操作説明を表示
if not st.session_state.intro_shown:
    with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
        st.markdown("こちらは生成AIによる音声英会話の練習アプリです。何度も繰り返し練習し、英語力をアップさせましょう。")
        st.markdown("**【操作説明】**")
        st.success("""
- モードと再生速度を選択し、「開始」ボタンで練習を始めます。
- モードは「日常英会話」「シャドーイング」「ディクテーション」から選べます。
- 発話後、5秒間沈黙することで音声入力が完了します。
- 「一時中断」ボタン相当の挙動は、開始ボタンを再押下しないことで代替できます。
""")
    st.session_state.intro_shown = True
st.divider()

# ==============================
# メッセージ履歴の表示
# ==============================
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(message["content"])
    elif message["role"] == "user":
        with st.chat_message("user", avatar=ct.USER_ICON_PATH):
            st.markdown(message["content"])
    else:
        st.divider()

# 実行補助ボタン（必要に応じて）
if st.session_state.shadowing_flg:
    st.session_state.shadowing_button_flg = st.button("シャドーイング開始")
if st.session_state.dictation_flg:
    st.session_state.dictation_button_flg = st.button("ディクテーション開始")

# ディクテーション入力案内
if st.session_state.chat_open_flg:
    st.info("AIが読み上げた音声を、画面下部のチャット欄からそのまま入力・送信してください。")

st.session_state.dictation_chat_message = st.chat_input("※「ディクテーション」選択時以外は送信不可")
if st.session_state.dictation_chat_message and not st.session_state.chat_open_flg:
    st.stop()

# ==============================
# 「開始」ボタン押下後の処理
# ==============================
if st.session_state.start_flg:

    # ---------- モード：ディクテーション ----------
    if st.session_state.mode == ct.MODE_3 and (
        st.session_state.dictation_button_flg
        or st.session_state.dictation_count == 0
        or st.session_state.dictation_chat_message
    ):
        if st.session_state.dictation_first_flg:
            st.session_state.chain_create_problem = ft.create_chain(ct.SYSTEM_TEMPLATE_CREATE_PROBLEM)
            st.session_state.dictation_first_flg = False

        # まだ入力を受け付けていない＝問題文の提示フェーズ
        if not st.session_state.chat_open_flg:
            with st.spinner('問題文生成中...'):
                st.session_state.problem, _ = ft.create_problem_and_play_audio()

            # 生成した英文を即表示（可視化）
            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(st.session_state.problem)
            st.session_state.messages.append(
                {"role": "assistant", "content": st.session_state.problem}
            )

            # 入力を受け付ける状態に遷移（rerunは行わず、ここで一旦停止して入力待ち）
            st.session_state.chat_open_flg = True
            st.session_state.dictation_flg = False
            st.stop()

        # 回答入力後の評価フェーズ
        else:
            if not st.session_state.dictation_chat_message:
                st.stop()

            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(st.session_state.problem)
            with st.chat_message("user", avatar=ct.USER_ICON_PATH):
                st.markdown(st.session_state.dictation_chat_message)

            st.session_state.messages.append({"role": "assistant", "content": st.session_state.problem})
            st.session_state.messages.append({"role": "user", "content": st.session_state.dictation_chat_message})

            with st.spinner('評価結果の生成中...'):
                system_template = ct.SYSTEM_TEMPLATE_EVALUATION.format(
                    llm_text=st.session_state.problem,
                    user_text=st.session_state.dictation_chat_message
                )
                st.session_state.chain_evaluation = ft.create_chain(system_template)
                llm_response_evaluation = ft.create_evaluation()

            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(llm_response_evaluation)
            st.session_state.messages.append({"role": "assistant", "content": llm_response_evaluation})
            st.session_state.messages.append({"role": "other"})

            st.session_state.dictation_flg = True
            st.session_state.dictation_chat_message = ""
            st.session_state.dictation_count += 1
            st.session_state.chat_open_flg = False
            st.rerun()

    # ---------- モード：日常英会話 ----------
    if st.session_state.mode == ct.MODE_1:
        audio_input_file_path = f"{ct.AUDIO_INPUT_DIR}/audio_input_{int(time.time())}.wav"
        audio_input_file_path = audio_input_file_path.replace("}", "")  # 万一の波括弧混入回避
        ft.record_audio(audio_input_file_path)

        with st.spinner('音声入力をテキストに変換中...'):
            transcript = ft.transcribe_audio(audio_input_file_path)
            audio_input_text = transcript.text

        with st.chat_message("user", avatar=ct.USER_ICON_PATH):
            st.markdown(audio_input_text)

        with st.spinner("回答の音声読み上げ準備中..."):
            llm_response = st.session_state.chain_basic_conversation.predict(input=audio_input_text)
            resp = st.session_state.openai_obj.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=llm_response
            )
            audio_bytes = getattr(resp, "content", resp)
            audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
            ft.save_to_wav(audio_bytes, audio_output_file_path)

        ft.play_wav(audio_output_file_path, speed=st.session_state.speed)

        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(llm_response)

        st.session_state.messages.append({"role": "user", "content": audio_input_text})
        st.session_state.messages.append({"role": "assistant", "content": llm_response})

    # ---------- モード：シャドーイング ----------
    if st.session_state.mode == ct.MODE_2 and (
        st.session_state.shadowing_button_flg
        or st.session_state.get("shadowing_in_progress", False)
        or st.session_state.shadowing_count == 0
    ):
        # 初回チェーン
        if st.session_state.shadowing_first_flg:
            st.session_state.chain_create_problem = ft.create_chain(ct.SYSTEM_TEMPLATE_CREATE_PROBLEM)
            st.session_state.shadowing_first_flg = False

        # 進行中でなければ「問題決定→TTS再生→録音へ」
        if not st.session_state.shadowing_in_progress:
            custom_sentence = st.text_input(
                "読み上げたい英文を入力（空欄ならAIが自動生成します）",
                placeholder="例：It's a beautiful day today, isn't it?"
            )

            if "show_text_flg" not in st.session_state:
                st.session_state.show_text_flg = True
            st.session_state.show_text_flg = st.checkbox(
                "英文を表示する（ONで見ながら練習）",
                value=st.session_state.show_text_flg,
                key="show_text_checkbox"
            )

            if custom_sentence:
                st.session_state.problem = custom_sentence
            else:
                with st.spinner('問題文生成中...'):
                    # ★ 三重再生解消：ここでは再生しない版を使用
                    st.session_state.problem = ft.generate_problem_only()

            # 評価対象を固定
            st.session_state.shadowing_eval_target = st.session_state.problem

            # 表示（履歴は重複登録回避）
            if st.session_state.show_text_flg:
                with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                    st.markdown(f"**練習文:** {st.session_state.problem}")
            if not st.session_state.messages or st.session_state.messages[-1].get("content") != f"**練習文:** {st.session_state.problem}":
                st.session_state.messages.append({"role": "assistant", "content": f"**練習文:** {st.session_state.problem}"})

            # TTS 2回再生（聞き取り→シャドーイング）
            resp1 = st.session_state.openai_obj.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=st.session_state.problem
            )
            audio_bytes1 = getattr(resp1, "content", resp1)
            audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
            ft.save_to_wav(audio_bytes1, audio_output_file_path)

            st.info("🔊【1回目】聞き取り練習中...")
            ft.play_wav(audio_output_file_path, st.session_state.speed, keep_file=True)
            st.info("🗣️【2回目】シャドーイング練習スタート！")
            ft.play_wav(audio_output_file_path, st.session_state.speed, keep_file=False)

            # 録音フェーズへ
            st.session_state.shadowing_in_progress = True

        # 録音→文字起こし（進行中）
        if st.session_state.shadowing_in_progress:
            audio_input_file_path = f"{ct.AUDIO_INPUT_DIR}/audio_input_{int(time.time())}.wav"
            audio_input_file_path = audio_input_file_path.replace("}", "")
            ft.record_audio(audio_input_file_path)

            with st.spinner('音声入力をテキストに変換中...'):
                transcript = ft.transcribe_audio(audio_input_file_path)
                audio_input_text = transcript.text

            if not audio_input_text.strip():
                st.warning("音声が検出されませんでした。もう一度お試しください。")
                st.stop()

            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(st.session_state.shadowing_eval_target)
            with st.chat_message("user", avatar=ct.USER_ICON_PATH):
                st.markdown(audio_input_text)

            st.session_state.messages.append({"role": "user", "content": audio_input_text})

            with st.spinner('評価結果の生成中...'):
                system_template = ct.SYSTEM_TEMPLATE_EVALUATION.format(
                    llm_text=st.session_state.shadowing_eval_target,
                    user_text=audio_input_text
                )
                st.session_state.chain_evaluation = ft.create_chain(system_template)
                llm_response_evaluation = ft.create_evaluation()

            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(llm_response_evaluation)

            st.session_state.messages.append({"role": "assistant", "content": llm_response_evaluation})
            st.session_state.messages.append({"role": "other"})

            # 後片付け：次の問題はユーザーが再度「開始」押下 or ボタン押下で
            st.session_state.shadowing_eval_target = ""
            st.session_state.shadowing_in_progress = False
            st.session_state.shadowing_flg = True
            st.session_state.shadowing_count += 1

            st.rerun()
