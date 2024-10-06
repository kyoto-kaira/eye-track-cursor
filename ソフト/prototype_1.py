import sys
from PyQt5.QtCore import QSize, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox
from ui_gui_for_line_of_sight import Ui_pio_dialog
from first_screen import Ui_FirstScreen  # first_screen.py に変換されたUIファイルをインポート
from second_screen import Ui_SecondScreen  # second_screen.py に変換されたUIファイルをインポート

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_SecondScreen()
        self.ui.setupUi(self)
#        self.ui.btn_left_up.clicked.connect(self.test)
        # QStackedWidgetを作成して画面を切り替え可能にする
        self.stacked_widget = QStackedWidget()

        # 最初の画面を設定
        self.first_screen = Ui_FirstScreen()
        self.first_widget = QMainWindow()
        self.first_screen.setupUi(self.first_widget)

        # 2つ目の画面を設定
        self.second_screen = Ui_SecondScreen()
        self.second_widget = QMainWindow()
        self.second_screen.setupUi(self.second_widget)

        # QStackedWidgetに各画面を追加
        self.stacked_widget.addWidget(self.first_widget)
        self.stacked_widget.addWidget(self.second_widget)

        # 画面サイズを 1024x800 に設定
        self.setFixedSize(1024, 768)

        # 最初の画面を表示
        self.setCentralWidget(self.stacked_widget)
        self.show_first_screen()

        # ボタンのサイズを設定（必要に応じて変更）
    #    self.first_screen.pushButton.setFixedSize(80, 20)  # 最初の画面のボタンサイズ
    #    self.second_screen.pushButton.setFixedSize(80, 20)  # 2つ目の画面のボタンサイズ        

        # ボタンが押されたときのイベントを設定
        self.first_screen.pushButton.clicked.connect(self.show_second_screen)  # ボタンの名前に応じて変更
        self.second_screen.pushButton.clicked.connect(self.show_first_screen)  # ボタンの名前に応じて変更
        # 各画面の操作ボタンに機能を割り当てる
      #  self.first_screen.actionButton.clicked.connect(self.action_first_screen)  # 1つ目の画面のボタン
        self.second_screen.btn_left_up.clicked.connect(self.btn_left_up_cal)  # 2つ目の画面のボタン
        self.second_screen.btn_right_up.clicked.connect(self.btn_right_up_cal)  # 2つ目の画面のボタン
        self.second_screen.btn_right_down.clicked.connect(self.btn_right_down_cal)  # 2つ目の画面のボタン
        self.second_screen.btn_left_down.clicked.connect(self.btn_left_down_cal)  # 2つ目の画面のボタン




    def show_first_screen(self):
        # 最初の画面に切り替え
        self.stacked_widget.setCurrentWidget(self.first_widget)
    def show_second_screen(self):
        # 2つ目の画面に切り替え
        self.stacked_widget.setCurrentWidget(self.second_widget)
#キャリブレーション
    def btn_left_up_cal(self):
        QMessageBox.information(self, "左上", "左上のキャリブレーションします。")        
        self.second_screen.btn_left_up.setEnabled(True)
    def btn_right_up_cal(self):
        QMessageBox.information(self, "右上", "右上のキャリブレーションします。")        
        self.second_screen.btn_right_up.setEnabled(True)
    def btn_right_down_cal(self):
        QMessageBox.information(self, "右下", "右下のキャリブレーションします。")        
        self.second_screen.btn_right_down.setEnabled(True)
    def btn_left_down_cal(self):
        QMessageBox.information(self, "左下", "左下のキャリブレーションします。")        
        self.second_screen.btn_left_down.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Screen Switcher")
    window.show()
    sys.exit(app.exec_())
