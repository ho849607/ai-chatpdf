def chat_interface():
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**👤 사용자:** {chat['message']}")
        else:
            st.markdown(f"**🤖 GPT:** {chat['message']}")

    if st.session_state.lang == 'korean':
        user_chat_input = st.text_input("메시지를 입력하세요:", key="user_chat_input")
    else:
        user_chat_input = st.text_input("Enter your message:", key="user_chat_input")

    if user_chat_input:
        add_chat_message("user", user_chat_input)
        with st.spinner("GPT가 응답 중입니다..."):
            gpt_response = ask_gpt_question(user_chat_input, st.session_state.lang)
            add_chat_message("assistant", gpt_response)
            st.markdown(f"**🤖 GPT:** {gpt_response}")

