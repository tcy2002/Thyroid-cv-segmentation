from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

from growth.growth import *
from unet.processor import *
from watershed.watershed import *


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.width = 1500
        self.height = 840

        self.setWindowTitle("甲状腺前景分割")
        self.setFixedSize(self.width, self.height)

        self.img_selected = ""
        self.out_dirname = "./outputs"
        self.processing = False
        self.seeds = []
        self.seeds_bg = []
        self.method = "growth"

        self.growth = Growth()
        self.unet = Processor()
        self.unet.extract('./unet/model/30.pth')
        self.watershed = WatershedSegmenter()

        # 创建菜单栏
        self.menu = self.menuBar()
        # 1. 文件
        self.file_menu = self.menu.addMenu("文件")
        # 1.1 导入图片
        self.load_file_action = QAction("导入图片", self)
        self.load_file_action.setShortcut("Ctrl+O")
        self.load_file_action.triggered.connect(self.slot_load_file)
        self.file_menu.addAction(self.load_file_action)
        # 1.2 导入文件夹
        self.load_directory_action = QAction("导入文件夹", self)
        self.load_directory_action.setShortcut("Ctrl+D")
        self.load_directory_action.triggered.connect(self.slot_load_directory)
        self.file_menu.addAction(self.load_directory_action)
        # 1.3 保存路径
        self.output_directory_action = QAction("选择保存路径", self)
        self.output_directory_action.setShortcut("Ctrl+S")
        self.output_directory_action.triggered.connect(self.slot_output_directory)
        self.file_menu.addAction(self.output_directory_action)
        # 2. 编辑
        self.edit_menu = self.menu.addMenu("编辑")
        # 2.1 设置分割方法
        self.set_method_menu = self.edit_menu.addMenu("分割方法")
        # 2.1.1 区域生长
        self.growth_action = QAction("区域生长", self)
        self.growth_action.setCheckable(True)
        self.growth_action.setChecked(True)
        self.growth_action.triggered.connect(lambda: self.slot_set_method("growth"))
        self.set_method_menu.addAction(self.growth_action)
        # 2.1.2 分水岭
        self.watershed_action = QAction("分水岭", self)
        self.watershed_action.setCheckable(True)
        self.watershed_action.triggered.connect(lambda: self.slot_set_method("watershed"))
        self.set_method_menu.addAction(self.watershed_action)
        # 2.1.3 UNet
        self.unet_action = QAction("UNet", self)
        self.unet_action.setCheckable(True)
        self.unet_action.triggered.connect(lambda: self.slot_set_method("unet"))
        self.set_method_menu.addAction(self.unet_action)
        # 选项互斥
        self.set_method_group = QActionGroup(self)
        self.set_method_group.addAction(self.growth_action)
        self.set_method_group.addAction(self.watershed_action)
        self.set_method_group.addAction(self.unet_action)
        self.set_method_group.setExclusive(True)
        # 3. 帮助
        self.help_menu = self.menu.addMenu("帮助")
        self.help_action = QAction("操作说明", self)
        self.help_action.setShortcut("Ctrl+H")
        self.help_action.triggered.connect(self.slot_help)
        self.help_menu.addAction(self.help_action)

        # 生成布局
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # 左侧文件列表：上方为原图片，下方为处理后的图片
        self.list_widget = QWidget()
        self.list_widget.setMinimumWidth(200)
        self.list_layout = QVBoxLayout()
        self.list_widget.setLayout(self.list_layout)
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(lambda item: self.slot_show_image(item, True))
        self.output_list_widget = QListWidget()
        self.output_list_widget.itemClicked.connect(lambda item: self.slot_show_image(item, False))
        self.list_layout.addWidget(self.file_list_widget)
        self.list_layout.addWidget(self.output_list_widget)
        self.main_layout.addWidget(self.list_widget)

        # 中间显示图片
        self.image_label = QLabel()
        self.image_label.mousePressEvent = self.slot_mouse_press
        self.image_label.setFixedSize(1024, 768)
        self.main_layout.addWidget(self.image_label)

        # 右侧控制面板
        self.control_widget = QWidget()
        self.control_widget.setMinimumWidth(160)
        self.control_layout = QVBoxLayout()
        self.control_layout.setAlignment(Qt.AlignTop)
        self.control_widget.setLayout(self.control_layout)
        self.main_layout.addWidget(self.control_widget)

        # 控制面板的按钮
        self.button1 = QPushButton("标记种子点")
        self.button1.clicked.connect(self.slot_button1_clicked)
        self.control_layout.addWidget(self.button1)
        self.button2 = QPushButton("开始分割")
        self.button2.clicked.connect(self.slot_button2_clicked)
        self.control_layout.addWidget(self.button2)
        self.button3 = QPushButton("批量分割")
        self.button3.clicked.connect(self.slot_button3_clicked)
        self.control_layout.addWidget(self.button3)
        self.button4 = QPushButton("清空工作区")
        self.button4.clicked.connect(self.slot_button4_clicked)
        self.control_layout.addWidget(self.button4)

    def slot_load_file(self):
        file_names, file_type = QFileDialog.getOpenFileNames(self, "打开文件", "./", "Image Files(*.png *.jpg *.bmp)")
        if len(file_names) == 0:
            return

        for file_name in file_names:
            print("导入图片：", file_name)
            new_item = QListWidgetItem(file_name)
            self.file_list_widget.addItem(new_item)
            self.slot_show_image(new_item, True)

    def slot_load_directory(self):
        dirname = QFileDialog.getExistingDirectory(self, "选择文件夹", "./")
        if dirname == "":
            return

        print("导入文件夹：", dirname)
        files = os.listdir(dirname)
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".bmp"):
                new_item = QListWidgetItem(os.path.join(dirname, file))
                self.file_list_widget.addItem(new_item)
                self.slot_show_image(new_item, True)

    def slot_output_directory(self):
        dirname = QFileDialog.getExistingDirectory(self, "选择保存路径", "./")
        if dirname == "":
            return

        print("保存路径：", dirname)
        self.out_dirname = dirname

    def slot_set_method(self, method):
        print("设置分割方法：", method)
        self.method = method
        self.button1.setEnabled(method != "unet")

    def slot_show_image(self, item, select):
        if self.processing:
            self.processing = False
            self.seeds.clear()

        print("显示图片：", item.text())
        self.image_label.setPixmap(QPixmap(item.text()))
        if select:
            self.img_selected = item.text()
        else:
            self.img_selected = ""

    def slot_mouse_press(self, event):
        """
        鼠标点击事件，显示点击位置的像素值
        """
        if self.img_selected == "" or not self.processing:
            return

        button = event.button()
        if button != Qt.LeftButton and button != Qt.RightButton:
            return

        img = self.image_label.pixmap().toImage()
        pos = event.pos()
        if pos.x() < 0 or pos.y() < 0 or pos.x() >= img.width() or pos.y() >= img.height():
            return

        # 在展示窗中绘制点
        painter = QPainter()
        painter.begin(img)
        if self.method == 'growth' or button == Qt.LeftButton:
            painter.setPen(QPen(Qt.white, 5))
        else:
            painter.setPen(QPen(Qt.yellow, 5))
        painter.drawPoint(pos)
        painter.end()
        self.image_label.setPixmap(QPixmap.fromImage(img))

        print("点击位置：", pos.x(), pos.y(), button)
        # 若为左键点击，则添加种子点，否则添加背景点
        if self.method == 'growth' or button == Qt.LeftButton:
            self.seeds.append((pos.x(), pos.y()))
        else:
            self.seeds_bg.append((pos.x(), pos.y()))

    def slot_button1_clicked(self):
        """
        点击事件：标记种子点
        """
        if self.img_selected == "":
            QMessageBox.information(self, "提示", "请先选择图片", QMessageBox.Ok)
            return

        print("开始选择种子点")
        self.processing = True
        self.image_label.setCursor(Qt.CrossCursor)

    def slot_button2_clicked(self):
        """
        点击事件：开始分割
        """
        if self.img_selected == "":
            QMessageBox.information(self, "提示", "请先选择图片", QMessageBox.Ok)
            return
        if self.method != 'unet' and len(self.seeds) == 0:
            QMessageBox.information(self, "提示", "请先标记种子点", QMessageBox.Ok)
            return
        if self.out_dirname == "":
            QMessageBox.information(self, "提示", "请先选择保存路径", QMessageBox.Ok)
            return

        self.seg_processing()

    def slot_button3_clicked(self):
        """
        点击事件：批量分割
        """
        if self.file_list_widget.count() == 0:
            QMessageBox.information(self, "提示", "请先导入图片", QMessageBox.Ok)
            return
        if self.method != 'unet':
            ans = QMessageBox.warning(self, "警告", "当前种子点将应用到所有分割，是否继续", QMessageBox.Ok, QMessageBox.No)
            if ans == QMessageBox.No:
                return
            if len(self.seeds) == 0:
                QMessageBox.information(self, "提示", "请先标记种子点", QMessageBox.Ok)
                return
        if self.out_dirname == "":
            QMessageBox.information(self, "提示", "请先选择保存路径", QMessageBox.Ok)
            return

        for i in range(self.file_list_widget.count()):
            self.img_selected = self.file_list_widget.item(i).text()
            self.seg_processing()

    def slot_button4_clicked(self):
        """
        点击事件：清空
        """
        self.file_list_widget.clear()
        self.output_list_widget.clear()
        self.image_label.clear()
        self.img_selected = ""
        self.out_dirname = "./outputs"
        self.seeds = []
        self.processing = False

    def seg_processing(self):
        """
        分割计算
        """
        print("开始分割计算：", self.img_selected)
        self.processing = False
        self.image_label.setCursor(Qt.ArrowCursor)
        name = os.path.basename(self.img_selected)

        if self.method == 'growth':
            name = "growth_" + name
            out_path = os.path.join(self.out_dirname, name)
            self.seeds.extend(self.seeds_bg)
            self.growth.grow(self.seeds, self.img_selected, out_path)
        elif self.method == 'watershed':
            name = "watershed_" + name
            out_path = os.path.join(self.out_dirname, name)
            self.watershed.segmentation(self.seeds, self.seeds_bg, self.img_selected, out_path)
        elif self.method == 'unet':
            name = "unet_" + name
            out_path = os.path.join(self.out_dirname, name)
            self.unet.predict(self.img_selected, out_path)
        else:
            return

        new_item = QListWidgetItem(out_path)
        self.output_list_widget.addItem(new_item)
        self.image_label.setPixmap(QPixmap(out_path))
        self.img_selected = ""
        self.seeds.clear()
        self.seeds_bg.clear()
        print("分割计算结束：", out_path)

    def slot_help(self):
        QMessageBox.information(self, "操作说明", "主菜单说明：\n"
                                              "1.文件/导入图片：导入单张图片\n"
                                              "2.文件/导入文件夹：导入文件夹中所有图片\n"
                                              "3.文件/选择保存路径：选择保存路径（默认：./outputs）\n"
                                              "4.编辑/分割方法：选择3种不同的分割方法\n"
                                              "\n"
                                              "右侧按钮说明：\n"
                                              "5.标记种子点：点击图片标记种子点，UNet方法无需此步骤\n"
                                              "6.开始分割：种子点标记结束后，开始分割当前图片\n"
                                              "7.批量分割：种子点标记结束后，开始分割所有图片\n"
                                              "8.清空工作区：清空工作区中所有图片、种子点以及分割结果，不会删除已经保存的文件\n"
                                              "\n"
                                              "关于种子点标记：\n"
                                              "9.区域增长：允许标记多个种子点，所有种子点均为前景区域\n"
                                              "10.分水岭：允许标记多个种子点，鼠标左键标注的种子点为前景点，鼠标右键标注的种子点为背景点。\n", QMessageBox.Ok)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
