# Generated by Django 2.2.5 on 2019-10-13 18:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('calendarapp', '0003_event_owner'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='event',
            name='url',
        ),
        migrations.AddField(
            model_name='event',
            name='place',
            field=models.CharField(default='', max_length=100),
        ),
        migrations.AlterField(
            model_name='event',
            name='title',
            field=models.CharField(default='', max_length=100),
        ),
    ]
