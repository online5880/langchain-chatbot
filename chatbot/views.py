import os
import random
from django.shortcuts import render
from .models import QuestionAnswer
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
import re
from transformers import pipeline
from .key import API_KEY
from .sentiment import label_mapping,get_sentiment_feedback

model_id = 'hun3359/klue-bert-base-sentiment'
model_pipeline = pipeline("text-classification", model=model_id)

os.environ["OPENAI_API_KEY"] = (API_KEY)
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# LangChain 메모리 설정
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# LangChain을 사용할 OpenAI 모델 및 설정
llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o-mini', verbose=False)

# 피드백 템플릿 설정
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 초등학생 친구들의 꿈을 응원하는 착한 로봇이야. 친구들이 좋아하는 것과 잘하는 것을 찾아주고, \
                재미있는 직업들을 소개해줘. 어려운 말은 쓰지 말고, 친구처럼 쉽고 재미있게 이야기해줘. \
                친구들의 꿈을 크게 키워주고 용기를 주는 멋진 응원을 해줘! 그리고 {sentiment}에 맞게 반응해줘."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "질문: {question}\n친구의 대답: {user_answer}\n \
               친구가 한 대답을 보고 좋아하는 것과 잘하는 것을 찾아줘. \
               그리고 그것과 관련된 신나는 직업들을 소개해주고, 친구의 꿈을 응원해줘. \
               어려운 말은 쓰지 말고, 초등학생 친구가 쉽게 이해할 수 있게 설명해줘.")
])

# 세션 기록을 저장할 딕셔너리
store = {}
session_id = "abc"

# 세션 기록 관리
def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()

    # 히스토리의 길이를 20개로 제한
    if len(store[session_ids].messages) > 20:
        store[session_ids].messages = store[session_ids].messages[-20:]

    return store[session_ids]

# LangChain을 통해 피드백 생성 체인 결합
complex_chain = prompt | llm

# RunnableWithMessageHistory 생성
chain_with_history = RunnableWithMessageHistory(
    complex_chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",
    history_messages_key="chat_history",
)

# **텍스트**를 <strong>태그로 변환하는 함수
def format_text_with_bold(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

# OpenAI 피드백 생성 함수
def generate_feedback_openai(question, user_answer,sentiment):
    try:
        input_data = {
            "question": question,
            "sentiment":sentiment,
            "user_answer": user_answer[:1000],  # 최대 1000자까지 제한
            "input": f"질문: {question}\n학생의 답변: {user_answer[:1000]}"
        }

        # LangChain을 통한 피드백 생성
        feedback_text = chain_with_history.invoke(
            input_data,
            config={"configurable": {"session_id": session_id, "max_tokens": 500}},
        )

        # 감정 분석 결과에 따라 피드백 수정
        sentiment_feedback = get_sentiment_feedback(sentiment)

        # 피드백에 감정 결과를 추가
        final_feedback = format_text_with_bold(feedback_text.content) + f"<br><br>추가 피드백: {sentiment_feedback}"
        return final_feedback

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return "피드백을 생성하는 도중 오류가 발생했습니다. 나중에 다시 시도해 주세요."

# Django 챗봇 뷰 함수
def chatbot(request):
    # 새로고침 시 세션 초기화 (GET 요청도 포함)
    if request.method == "GET" and not request.GET.get('answer'):
        request.session.flush()  # 세션 데이터 전체 삭제
        request.session['chat_log'] = []  # 빈 채팅 로그로 초기화

    # 세션에서 대화 로그를 불러옴 (없으면 빈 리스트 생성)
    chat_log = request.session.get('chat_log', [])

    user_answer = request.POST.get('answer', '')  # 사용자가 입력한 답변
    random_question = None
    openai_feedback = ""

    # 모든 질문을 가져옴
    all_questions = QuestionAnswer.objects.all()

    if not user_answer:  # 사용자가 답변을 입력하지 않았을 때만 랜덤 질문 선택
        if all_questions.exists():
            # 무작위로 질문 하나 선택
            random_question = random.choice(all_questions)
            # 질문을 대화 로그에 추가
            chat_log.append({'role': 'bot', 'message': f"질문: {random_question.question}"})
        else:
            chat_log.append({'role': 'bot', 'message': "질문이 없습니다."})
    else:
        selected_question_id = request.POST.get('question_id', '')  # 사용자가 답변한 질문의 ID
        try:
            # 사용자가 답변하는 질문을 ID로 찾음
            question_instance = QuestionAnswer.objects.get(id=selected_question_id)

            # 사용자의 답변을 대화 로그에 추가
            chat_log.append({'role': 'user', 'message': user_answer})
            
            rst = model_pipeline([user_answer])
            sentiment_label = rst[0].get('label', '')
            mapped_label = label_mapping.get(sentiment_label)

            # Langchain을 통해 피드백 생성
            openai_feedback = generate_feedback_openai(
                question=question_instance.question,
                user_answer=user_answer.strip(),
                sentiment=mapped_label
            )

            # 챗봇의 피드백을 대화 로그에 추가
            chat_log.append({'role': 'bot', 'message': openai_feedback})

            # 같은 질문을 다시 보여줌
            random_question = question_instance

        except QuestionAnswer.DoesNotExist:
            chat_log.append({'role': 'bot', 'message': "해당 질문을 찾을 수 없습니다."})

    # 세션에 대화 로그 저장
    request.session['chat_log'] = chat_log

    return render(request, 'chatbot/chatbot.html', {
        'chat_log': chat_log,  # 전체 대화 로그 전달
        'random_question': random_question,
        'openai_feedback': openai_feedback
    })