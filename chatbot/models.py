from django.db import models

class QuestionAnswer(models.Model):
    gender = models.CharField(max_length=10)
    category_name = models.CharField(max_length=100, blank=True, null=True)
    question = models.TextField()  # 질문 필드
    answer_1 = models.TextField(blank=True, null=True)  # 답변 필드들
    answer_2 = models.TextField(blank=True, null=True)
    answer_3 = models.TextField(blank=True, null=True)
    answer_4 = models.TextField(blank=True, null=True)
    answer_5 = models.TextField(blank=True, null=True)
    answer_6 = models.TextField(blank=True, null=True)
    grade = models.CharField(max_length=10)
    comment = models.TextField()  # 코멘트 (답변 후 피드백)

    def __str__(self):
        return self.question
