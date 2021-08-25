
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from bs4 import BeautifulSoup

code_dir = os.path.dirname(os.path.abspath(__file__))


class WelcomeMessage(QtWidgets.QDialog):
    def __init__(self, cache_fname, html_fname, has_file=False, parent=None):
        super(WelcomeMessage, self).__init__(parent)
        with open(html_fname) as html_file:
            html = BeautifulSoup(''.join(html_file.readlines()), 'html.parser')

        title_tag = html.find('title')
        title = title_tag.text.strip()

        divisions = html.find_all('div')
        no_file_msg = ''
        main_msg = ''
        for div in divisions:
            if 'class' in div.attrs:
                div_classes = div.attrs['class']
                if 'no-file' in div_classes:
                    raw_text = div.decode_contents()
                    no_file_msg = raw_text.replace('\n', '').replace('\t', '')
                elif 'main' in div_classes:
                    raw_text = div.decode_contents()
                    main_msg = raw_text.replace('\n', '').replace('\t', '')


        self.setWindowTitle(title)
        self.cache_fname = cache_fname

        info_msg = """<h3>%s</h3>""" % title

        if not has_file:
            info_msg += no_file_msg

        info_msg += main_msg

        self.text = QtWidgets.QTextEdit(info_msg)
        self.text.setMinimumWidth(450)
        self.text.setMinimumHeight(300)
        self.text.setReadOnly(True)

        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.clicked.connect(self.ok)

        self.tick_box = QtWidgets.QCheckBox()
        self.tick_box.setChecked(True)

        main_layout = QtWidgets.QVBoxLayout()
        bottom_row = QtWidgets.QHBoxLayout()

        bottom_row.addWidget(QtWidgets.QLabel("Show this message on startup: "))
        bottom_row.addWidget(self.tick_box)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.ok_button)

        main_layout.addWidget(self.text)
        main_layout.addLayout(bottom_row)

        self.setLayout(main_layout)
        self.show()

    def ok(self, *args):
        state = self.tick_box.checkState()
        cache_file = os.path.join(code_dir, self.cache_fname)
        if state == 0:
            os.system("touch %s" % cache_file)
        elif state == 2 and os.path.exists(cache_file):
            os.system("rm %s" % cache_file)

        self.close()
