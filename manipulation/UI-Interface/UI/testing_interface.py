from PyQt5.QtCore    import *
from PyQt5.QtWidgets import *

class Page(QWidget):
    def __init__(self, parent=None):             # __init__
        super(Page, self).__init__(parent)       # __init__

        my_label = QLabel("This is my labet")
        layout   = QVBoxLayout()

        layout.addWidget(my_label)

        mainLayout = QGridLayout()
        mainLayout.addLayout(layout, 0, 1)

        self.setLayout(mainLayout)
        self.setWindowTitle("my first Qt app")

if __name__ == '__main__':                       #
    import sys
    print("LOEREE")
    app = QApplication(sys.argv)
    window = Page()
    window.show()
    sys.exit(app.exec_())