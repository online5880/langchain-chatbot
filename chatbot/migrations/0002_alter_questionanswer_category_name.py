# Generated by Django 5.1.2 on 2024-10-16 03:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='questionanswer',
            name='category_name',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]