# Generated by Django 4.1.10 on 2024-05-02 23:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('IR', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='documentModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('document', models.TextField(blank=True, null=True)),
            ],
        ),
    ]
