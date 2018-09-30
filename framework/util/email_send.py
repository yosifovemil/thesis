#!/usr/bin/env python
# -*- coding: utf-8 -*-

import smtplib
from email.mime.text import MIMEText
import os


def send_email(txt):
    mail = 'eey9@aber.ac.uk'
    f = open(os.path.join(os.environ['HOME'], '.email'), 'r')
    password = f.readline().strip()

    message = MIMEText(txt)
    message['Subject'] = txt
    message['From'] = mail
    message['To'] = mail

    s = smtplib.SMTP('smtp.office365.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(mail, password)
    s.sendmail(mail, mail, message.as_string())
    s.quit()
