# Generated by Django 4.1.10 on 2024-05-01 02:24

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SearchEngineModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('query', models.CharField(max_length=500)),
                ('model', models.CharField(choices=[('Document Term Matrix OR Bitwise', 'Document Term Matrix OR Bitwise'), ('Document Term Matrix AND Bitwise', 'Document Term Matrix AND Bitwise'), ('Inverted Index OR', 'Inverted Index OR'), ('Inverted Index AND', 'Inverted Index AND'), ('TFID', 'TFID')], max_length=100)),
            ],
        ),
    ]